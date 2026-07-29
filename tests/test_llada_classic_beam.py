from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Config
from d5p4.diffusion_llada import (
    LLADASampler,
    leftmost_transfer_mask,
    left_to_right_beam_sample,
)
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample import get_subsample_selector


MASK_TOKEN_ID = 6
EOS_TOKEN_ID = 5
VOCAB_SIZE = 7


class RecordingMaskedLM(nn.Module):
    def __init__(self, logits_fn=None):
        super().__init__()
        self.config = SimpleNamespace(mask_token_id=MASK_TOKEN_ID, vocab_size=VOCAB_SIZE)
        self.logits_fn = logits_fn or self._default_logits
        self.calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    @staticmethod
    def _default_logits(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
        logits = torch.full((input_ids.shape[0], VOCAB_SIZE), -4.0)
        prompt_signal = input_ids[:, 0].remainder(2)
        logits[:, 0] = 3.0 + prompt_signal
        logits[:, 1] = 2.0 - prompt_signal
        logits[:, 2] = 1.0 + 0.2 * pos
        return logits

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        return_dict,
        output_hidden_states,
        last_hidden_state_only,
        logits_slice,
    ):
        assert return_dict
        assert not output_hidden_states
        assert last_hidden_state_only
        assert logits_slice.stop == logits_slice.start + 1
        pos = logits_slice.start
        self.calls.append((input_ids.clone(), attention_mask.clone(), pos))
        logits = self.logits_fn(input_ids, pos)
        return SimpleNamespace(logits=logits.unsqueeze(1))


def _expected_score(model: RecordingMaskedLM, sequence: torch.Tensor, generation_start: int) -> float:
    """Mean log prob per generated token, i.e. what the sampler reports for a beam."""
    score = 0.0
    for pos in range(generation_start, sequence.shape[0]):
        partial = sequence.clone()
        partial[pos:] = MASK_TOKEN_ID
        logits = model.logits_fn(partial.unsqueeze(0), pos)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        score += float(log_probs[0, sequence[pos]])
    return score / (sequence.shape[0] - generation_start)


def test_classic_beam_preserves_committed_prefix_masks_suffix_and_sums_scores():
    model = RecordingMaskedLM()
    prompt = torch.tensor([[7, 8]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    sequences, scores, forwards = left_to_right_beam_sample(
        model,
        prompt,
        attention_mask,
        generation_length=3,
        beam_size=3,
        branching_factor=2,
    )

    assert sequences.shape == (1, 3, 5)
    assert scores.shape == (1, 3)
    assert forwards == 3
    assert torch.equal(sequences[0, :, :2], prompt.expand(3, -1))
    assert not torch.any(sequences[:, :, 2:] == MASK_TOKEN_ID)
    assert torch.all(scores[:, :-1] >= scores[:, 1:])

    for call_ids, call_attention, pos in model.calls:
        assert torch.equal(call_ids[:, :2], prompt.expand(call_ids.shape[0], -1))
        assert not torch.any(call_ids[:, 2:pos] == MASK_TOKEN_ID)
        assert torch.all(call_ids[:, pos:] == MASK_TOKEN_ID)
        assert torch.all(call_attention == 1)

    expected = torch.tensor(
        [_expected_score(model, sequence, generation_start=2) for sequence in sequences[0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(scores[0], expected)


def test_classic_beam_global_selection_can_take_two_children_from_one_parent():
    def logits_fn(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
        if pos == 1:
            logits = torch.full((input_ids.shape[0], VOCAB_SIZE), -20.0)
            logits[:, 0] = 0.0
            logits[:, 1] = -0.1
        else:
            parent_is_one = input_ids[:, 1] == 1
            logits = torch.zeros((input_ids.shape[0], VOCAB_SIZE))
            logits[parent_is_one] = -20.0
            logits[parent_is_one, 2] = 8.0
            logits[parent_is_one, 3] = 7.9
        return logits

    model = RecordingMaskedLM(logits_fn)
    sequences, _, _ = left_to_right_beam_sample(
        model,
        torch.tensor([[4]]),
        torch.ones((1, 1), dtype=torch.long),
        generation_length=2,
        beam_size=2,
        branching_factor=2,
    )

    assert sequences[0, :, 1].tolist() == [1, 1]
    assert set(sequences[0, :, 2].tolist()) == {2, 3}


def test_beam_size_one_matches_greedy_left_to_right():
    model = RecordingMaskedLM()
    prompt = torch.tensor([[9]], dtype=torch.long)
    sequences, scores, forwards = left_to_right_beam_sample(
        model,
        prompt,
        torch.ones_like(prompt),
        generation_length=4,
        beam_size=1,
    )

    greedy = torch.full((1, 5), MASK_TOKEN_ID, dtype=torch.long)
    greedy[:, :1] = prompt
    expected_score = 0.0
    for pos in range(1, 5):
        logits = model.logits_fn(greedy, pos)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_probs[:, MASK_TOKEN_ID] = -torch.inf
        token = log_probs.argmax(dim=-1)
        greedy[:, pos] = token
        expected_score += float(log_probs[0, token])

    assert torch.equal(sequences[:, 0], greedy)
    assert forwards == 4
    torch.testing.assert_close(scores, torch.tensor([[expected_score / 4]]))


def test_batched_and_single_example_classic_beam_match():
    prompts = torch.tensor([[8, 4], [9, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompts)

    batched_model = RecordingMaskedLM()
    batched_sequences, batched_scores, _ = left_to_right_beam_sample(
        batched_model,
        prompts,
        attention_mask,
        generation_length=3,
        beam_size=2,
    )

    single_results = [
        left_to_right_beam_sample(
            RecordingMaskedLM(),
            prompts[index : index + 1],
            attention_mask[index : index + 1],
            generation_length=3,
            beam_size=2,
        )
        for index in range(2)
    ]
    single_sequences = torch.cat([result[0] for result in single_results], dim=0)
    single_scores = torch.cat([result[1] for result in single_results], dim=0)

    assert torch.equal(batched_sequences, single_sequences)
    torch.testing.assert_close(batched_scores, single_scores)


def test_classic_beam_preserves_padded_prompt_attention_mask():
    model = RecordingMaskedLM()
    prompt = torch.tensor([[0, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1]], dtype=torch.long)

    sequences, _, _ = left_to_right_beam_sample(
        model,
        prompt,
        attention_mask,
        generation_length=2,
        beam_size=1,
    )

    assert torch.equal(sequences[0, 0, :2], prompt[0])
    for _, call_attention, _ in model.calls:
        assert call_attention.tolist() == [[0, 1, 1, 1]]


def _eos_preferring_logits(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
    del pos
    logits = torch.zeros((input_ids.shape[0], VOCAB_SIZE))
    logits[:, MASK_TOKEN_ID] = 100.0
    logits[:, EOS_TOKEN_ID] = 90.0
    return logits


def test_mask_token_is_excluded_and_generation_is_fixed_length_without_eos_ids():
    sequences, _, forwards = left_to_right_beam_sample(
        RecordingMaskedLM(_eos_preferring_logits),
        torch.tensor([[2]]),
        torch.ones((1, 1), dtype=torch.long),
        generation_length=3,
        beam_size=1,
    )

    assert sequences[0, 0, 1:].tolist() == [EOS_TOKEN_ID, EOS_TOKEN_ID, EOS_TOKEN_ID]
    assert MASK_TOKEN_ID not in sequences
    assert forwards == 3


def test_eos_stops_decoding_before_the_length_budget_is_used():
    sequences, scores, forwards = left_to_right_beam_sample(
        RecordingMaskedLM(_eos_preferring_logits),
        torch.tensor([[2]]),
        torch.ones((1, 1), dtype=torch.long),
        generation_length=5,
        beam_size=2,
        eos_token_ids=(EOS_TOKEN_ID,),
    )

    # Beam 0 emits EOS at the first generated position; beam 1 takes the runner-up token and
    # finishes one position later. Both are done after two forwards, well short of the budget of 5.
    assert forwards == 2
    assert sequences[0, 0, 1:].tolist() == [EOS_TOKEN_ID] * 5
    assert sequences[0, 1, 2:].tolist() == [EOS_TOKEN_ID] * 4
    assert MASK_TOKEN_ID not in sequences
    # Length 1, not 5: the padding after EOS contributes nothing to the score.
    expected = F.log_softmax(_eos_preferring_logits(torch.zeros(1, 1), 0).float(), dim=-1)[0, EOS_TOKEN_ID]
    torch.testing.assert_close(scores[0, 0], expected)


def test_finished_beam_keeps_exactly_one_slot_and_ranks_by_length_normalized_score():
    def logits_fn(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
        if pos == 1:
            # Beam 0 finishes immediately; beam 1 continues with token 0.
            logits = torch.full((input_ids.shape[0], VOCAB_SIZE), -20.0)
            logits[:, EOS_TOKEN_ID] = 0.0
            logits[:, 0] = -0.5
            return logits
        # Continuations are near-free, so the cumulative sums never cross over.
        logits = torch.full((input_ids.shape[0], VOCAB_SIZE), -20.0)
        logits[:, 1] = 0.0
        return logits

    sequences, scores, forwards = left_to_right_beam_sample(
        RecordingMaskedLM(logits_fn),
        torch.tensor([[3]]),
        torch.ones((1, 1), dtype=torch.long),
        generation_length=3,
        beam_size=2,
        eos_token_ids=(EOS_TOKEN_ID,),
    )

    assert forwards == 3
    # Beam order follows the cumulative sums used during search, so the finished hypothesis is
    # kept exactly once (not duplicated across the beam) and stays first.
    assert sequences[0, 0, 1:].tolist() == [EOS_TOKEN_ID, EOS_TOKEN_ID, EOS_TOKEN_ID]
    assert sequences[0, 1, 1:].tolist() == [0, 1, 1]
    # The reported scores are length-normalized, as in HuggingFace beam search with
    # length_penalty=1.0, so the three-token beam outranks the one-token beam despite the
    # worse cumulative sum. Consumers rank by score, not by beam index.
    assert scores[0, 1] > scores[0, 0]


def test_leftmost_transfer_mask_unmasks_in_position_order():
    mask_index = torch.tensor(
        [
            [False, True, True, True],
            [True, True, True, True],
        ],
    )
    counts = torch.tensor([2, 1])

    got = leftmost_transfer_mask(mask_index, counts)

    assert got.tolist() == [
        [False, True, True, False],
        [True, False, False, False],
    ]


def test_leftmost_transfer_mask_never_selects_decided_positions():
    mask_index = torch.tensor([[True, False, True, False, True]])
    got = leftmost_transfer_mask(mask_index, torch.tensor([3]))

    assert got.tolist() == [[True, False, True, False, True]]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"generation_length": 0, "beam_size": 1}, "generation_length"),
        ({"generation_length": 1, "beam_size": 0}, "beam_size"),
        ({"generation_length": 1, "beam_size": 1, "branching_factor": 0}, "branching_factor"),
        ({"generation_length": 1, "beam_size": 1, "branching_factor": VOCAB_SIZE}, "non-mask"),
        ({"generation_length": 3, "beam_size": 2, "branching_factor": 1}, "cannot populate"),
    ],
)
def test_classic_beam_rejects_invalid_sizes(kwargs, error):
    prompt = torch.tensor([[1]])
    with pytest.raises(ValueError, match=error):
        left_to_right_beam_sample(RecordingMaskedLM(), prompt, torch.ones_like(prompt), **kwargs)


def test_classic_beam_config_requires_conditional_ltr_beam_method():
    with pytest.raises(AssertionError, match="cfg_scale=1.0"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            cfg_scale=2.0,
            method="ltr_beam",
        )
    with pytest.raises(AssertionError, match="method=ltr_beam"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            cfg_scale=1.0,
            method="greedy_map",
        )
    with pytest.raises(AssertionError, match="logits_eos_inf"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            cfg_scale=1.0,
            method="ltr_beam",
            logits_eos_inf=True,
        )
    with pytest.raises(AssertionError, match="already left-to-right"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            cfg_scale=1.0,
            method="ltr_beam",
            force_left_to_right=True,
        )
    with pytest.raises(AssertionError, match="requires model=llada and llada_decoder=classic_beam"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="diffusion",
            method="ltr_beam",
        )


def test_forced_left_to_right_relaxes_nothing_else_in_llada_validation():
    config = Config(
        disable_sys_args=True,
        model="llada",
        force_left_to_right=True,
        gen_length=8,
        block_length=8,
        llada_steps=8,
    )

    assert config.force_left_to_right
    assert config.llada_decoder == "diffusion"


def test_llada_sampler_routes_classic_beam_and_reports_forward_count():
    config = Config(
        disable_sys_args=True,
        model="llada",
        llada_decoder="classic_beam",
        cfg_scale=1.0,
        method="ltr_beam",
        transversal=False,
        gen_length=3,
        n_groups=2,
        group_size=1,
    )
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.model = RecordingMaskedLM()
    sampler.distributed_utils = None
    sampler.mask_index = MASK_TOKEN_ID
    sampler.tokenizer = SimpleNamespace(eos_token_id=EOS_TOKEN_ID, get_added_vocab=dict)
    sampler._preprocess_prompt = lambda _prompt: torch.tensor([[1, 2]], dtype=torch.long)

    sequences, scores = sampler.sample("prompt", return_internal_scores=True)

    assert sequences.shape == (2, 5)
    assert scores.shape == (2,)
    assert sampler.last_forward_count == config.gen_length


def test_tiny_llada_model_runs_classic_beam_forward():
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        vocab_size=11,
        embedding_size=16,
        max_sequence_length=8,
        mask_token_id=10,
        rope=True,
        alibi=False,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        weight_tying=True,
        init_device="cpu",
    )
    model = LLaDAModelLM(config, init_params=True).eval()

    sequences, scores, forwards = left_to_right_beam_sample(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.long),
        generation_length=2,
        beam_size=2,
    )

    assert sequences.shape == (1, 2, 4)
    assert scores.shape == (1, 2)
    assert not torch.any(sequences[:, :, 2:] == config.mask_token_id)
    assert torch.isfinite(scores).all()


# ── partitioned beam search (llada_decoder="classic_beam" + transversal) ────────────────────


def test_single_group_partition_is_bit_identical_to_classic_beam():
    """The partitioned path must reduce exactly, so classic_beam results stay reproducible."""
    prompt = torch.tensor([[7, 8]], dtype=torch.long)
    attention = torch.ones_like(prompt)
    plain = left_to_right_beam_sample(
        RecordingMaskedLM(), prompt, attention, generation_length=3, beam_size=3, branching_factor=2,
    )
    grouped = left_to_right_beam_sample(
        RecordingMaskedLM(), prompt, attention, generation_length=3, beam_size=3, branching_factor=2,
        num_groups=1,
    )
    assert torch.equal(plain[0], grouped[0])
    torch.testing.assert_close(plain[1], grouped[1])
    assert plain[2] == grouped[2]


def test_partitioned_beam_seeds_each_group_from_a_different_token():
    """Groups only diverge because the first position is split across them by rank.

    Every beam starts holding the same prompt and all-[MASK] suffix, so per-group seeding would
    give identical groups forever. The split at the first generated position is what makes the
    partition mean anything.
    """
    prompt = torch.tensor([[9]], dtype=torch.long)
    groups, per_group = 3, 2
    sequences, scores, forwards = left_to_right_beam_sample(
        RecordingMaskedLM(), prompt, torch.ones_like(prompt),
        generation_length=4, beam_size=groups * per_group, branching_factor=3,
        num_groups=groups,
    )
    assert torch.isfinite(scores).all(), "every group must own a live hypothesis"
    group_leads = [row[0] for row in sequences[0, :, 1].view(groups, per_group).tolist()]
    assert len(set(group_leads)) == groups, f"groups share a lead token: {group_leads}"
    assert forwards == 4


def test_partition_changes_the_search_not_the_objective():
    """A partition can only lose likelihood: the global top-k is the unconstrained optimum."""
    prompt = torch.tensor([[9]], dtype=torch.long)
    kwargs = {"generation_length": 4, "beam_size": 6, "branching_factor": 3}
    _, global_scores, global_forwards = left_to_right_beam_sample(
        RecordingMaskedLM(), prompt, torch.ones_like(prompt), **kwargs,
    )
    _, split_scores, split_forwards = left_to_right_beam_sample(
        RecordingMaskedLM(), prompt, torch.ones_like(prompt), **kwargs, num_groups=3,
    )
    assert global_scores.max() >= split_scores.max()
    # Partitioning is free: identical forward count, so the arms stay compute-matched.
    assert global_forwards == split_forwards == kwargs["generation_length"]


def test_partitioned_beam_keeps_runner_up_prefixes_a_global_beam_would_drop():
    def logits_fn(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
        del pos
        # One dominant token, so a global beam piles onto its continuations while a partition is
        # forced to keep the runners-up alive inside their own groups.
        logits = torch.full((input_ids.shape[0], VOCAB_SIZE), -8.0)
        logits[:, 0] = 4.0
        logits[:, 1] = 1.0
        logits[:, 2] = 0.5
        return logits

    kwargs = {"generation_length": 3, "beam_size": 6, "branching_factor": 3}
    prompt = torch.tensor([[9]], dtype=torch.long)
    global_seqs, _, _ = left_to_right_beam_sample(
        RecordingMaskedLM(logits_fn), prompt, torch.ones_like(prompt), **kwargs,
    )
    split_seqs, _, _ = left_to_right_beam_sample(
        RecordingMaskedLM(logits_fn), prompt, torch.ones_like(prompt), **kwargs, num_groups=3,
    )
    global_leads = {row[0] for row in global_seqs[0, :, 1:].tolist()}
    split_leads = {row[0] for row in split_seqs[0, :, 1:].tolist()}
    assert len(split_leads) >= len(global_leads)
    assert len({tuple(row) for row in split_seqs[0, :, 1:].tolist()}) == kwargs["beam_size"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"generation_length": 2, "beam_size": 4, "num_groups": 0}, "num_groups"),
        ({"generation_length": 2, "beam_size": 4, "num_groups": 3}, "divisible"),
        ({"generation_length": 2, "beam_size": 4, "num_groups": 2, "branching_factor": 1}, "cannot populate"),
        # Only branching_factor candidates exist at the first position, and each group needs one.
        ({"generation_length": 4, "beam_size": 4, "num_groups": 4, "branching_factor": 2}, "cannot seed"),
    ],
)
def test_partitioned_beam_rejects_invalid_group_settings(kwargs, error):
    prompt = torch.tensor([[1]])
    with pytest.raises(ValueError, match=error):
        left_to_right_beam_sample(RecordingMaskedLM(), prompt, torch.ones_like(prompt), **kwargs)


def test_transversal_classic_beam_config_mirrors_the_d5p4_partition():
    config = Config(
        disable_sys_args=True,
        model="llada",
        llada_decoder="classic_beam",
        cfg_scale=1.0,
        method="ltr_beam",
        transversal=True,
        n_groups=3,
        group_size=3,
    )
    # n_groups beam groups of group_size beams each: the same 3x3 population as the D5P4 arm.
    assert config.batch_size == 9
    assert type(get_subsample_selector(config)).__name__ == "LTRBeamSelection"

    with pytest.raises(AssertionError, match="n_groups > 1"):
        Config(
            disable_sys_args=True, model="llada", llada_decoder="classic_beam", cfg_scale=1.0,
            method="ltr_beam", transversal=True, n_groups=1, group_size=9,
        )
    with pytest.raises(AssertionError, match="group_size > 1"):
        Config(
            disable_sys_args=True, model="llada", llada_decoder="classic_beam", cfg_scale=1.0,
            method="ltr_beam", transversal=True, n_groups=9, group_size=1,
        )
    with pytest.raises(AssertionError, match="batch_size"):
        Config(
            disable_sys_args=True, model="llada", llada_decoder="classic_beam", cfg_scale=1.0,
            method="ltr_beam", transversal=False, n_groups=1, group_size=1,
        )
    with pytest.raises(AssertionError, match="full global beam width in n_groups"):
        Config(
            disable_sys_args=True, model="llada", llada_decoder="classic_beam", cfg_scale=1.0,
            method="ltr_beam", transversal=False, n_groups=3, group_size=3,
        )


def test_llada_sampler_routes_transversal_classic_beam_with_group_partition():
    config = Config(
        disable_sys_args=True,
        model="llada",
        llada_decoder="classic_beam",
        cfg_scale=1.0,
        method="ltr_beam",
        gen_length=3,
        transversal=True,
        n_groups=2,
        group_size=2,
    )
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.model = RecordingMaskedLM()
    sampler.distributed_utils = None
    sampler.mask_index = MASK_TOKEN_ID
    sampler.tokenizer = SimpleNamespace(eos_token_id=EOS_TOKEN_ID, get_added_vocab=dict)
    sampler._preprocess_prompt = lambda _prompt: torch.tensor([[1, 2]], dtype=torch.long)

    sequences, scores = sampler.sample("prompt", return_internal_scores=True)

    assert sequences.shape == (4, 5)
    assert scores.shape == (4,)
    assert sampler.last_forward_count == config.gen_length

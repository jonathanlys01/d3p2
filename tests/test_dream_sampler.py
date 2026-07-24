from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import MaskedLMOutput

from d5p4.config import Config
from d5p4.diffusion_dream import DreamSampler, _localize_distributed_config
from d5p4.subsample import get_subsample_selector


class _FakeTokenizer:
    eos_token_id = 8
    unk_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert messages == [{"role": "user", "content": "question"}]
        assert add_generation_prompt
        assert not tokenize
        return "templated question"

    def __call__(self, prompts, add_special_tokens=False, padding=True, return_tensors="pt"):
        assert prompts == ["templated question"]
        assert not add_special_tokens
        assert padding
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return 7 if token == "<|im_end|>" else self.unk_token_id

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return ",".join(str(token_id) for token_id in token_ids)


class _ToyDreamModel(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids,
        attention_mask="full",
        return_dict=True,
        output_hidden_states=False,
        last_hidden_state_only=False,
        num_logits_to_keep=0,
    ):
        assert attention_mask == "full"
        assert return_dict
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        targets = (positions + 3) % self.vocab_size
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits.scatter_(-1, targets.view(1, -1, 1).expand(batch, -1, -1), 6.0)
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:]

        hidden = positions.float().view(1, seq_len, 1).expand(batch, -1, self.hidden_size)
        hidden_states = (hidden,) if output_hidden_states or last_hidden_state_only else None
        return MaskedLMOutput(logits=logits, hidden_states=hidden_states)


class _RecordingSelector(nn.Module):
    needs_embeddings = True
    distributed_utils = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cache_shapes = []

    def subsample(self, cache):
        self.cache_shapes.append(
            {
                "x": tuple(cache.x.shape),
                "log_p_x0": tuple(cache.log_p_x0.shape),
                "embeddings": tuple(cache.embeddings.shape),
            },
        )
        return torch.arange(self.config.n_groups) * self.config.group_size


def _build_sampler(config: Config) -> DreamSampler:
    object.__setattr__(config, "standalone_job", True)
    sampler = DreamSampler.__new__(DreamSampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.mask_index = 9
    sampler.max_position_embeddings = 64
    sampler.model = _ToyDreamModel()
    sampler.selector = get_subsample_selector(config)
    sampler.tokenizer = _FakeTokenizer()
    sampler.distributed_utils = None
    return sampler


def _config(**kwargs) -> Config:
    defaults = {
        "disable_sys_args": True,
        "model": "dream",
        "dream_steps": 3,
        "gen_length": 4,
        "n_groups": 2,
        "group_size": 1,
        "method": "baseline",
        "cat_temperature": 0.0,
        "dream_top_p": None,
        "dream_top_k": None,
        "subsample_start": 0,
        "subsample_end": 3,
        "standalone_job": True,
        "quiet": True,
    }
    defaults.update(kwargs)
    return Config(**defaults)


def test_dream_config_defaults_match_instruct_profile():
    config = Config(disable_sys_args=True, model="dream")

    assert config.embedding_dim == 3584
    assert config.dream_model_path == "Dream-org/Dream-v0-Instruct-7B"
    assert config.dream_steps == 256
    assert config.dream_alg == "entropy"
    assert config.dream_top_p == 0.9


def test_dream_forward_aligns_shifted_predictors_and_embeddings():
    sampler = _build_sampler(_config(gen_length=3, n_groups=1))
    x = torch.tensor([[1, 2, 9, 9, 9]])

    logits, embeddings = sampler._forward_model(x, need_embeddings=True)

    assert logits.argmax(dim=-1).tolist() == [[4, 5, 6]]
    assert embeddings is not None
    assert embeddings[0, :, 0].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("algorithm", ["origin", "maskgit_plus", "topk_margin", "entropy"])
def test_dream_all_reference_algorithms_fill_masks_and_preserve_prompt(algorithm):
    sampler = _build_sampler(_config(dream_alg=algorithm, cat_temperature=0.5))
    torch.manual_seed(4)

    samples, scores = sampler.sample("question", return_internal_scores=True)

    assert samples.shape == (2, 6)
    assert torch.equal(samples[:, :2], torch.tensor([[1, 2], [1, 2]]))
    assert not torch.any(samples[:, 2:] == sampler.mask_index)
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


def test_dream_top_k_filter_limits_sampling_support():
    sampler = _build_sampler(_config(dream_top_p=0.9, dream_top_k=1, cat_temperature=0.5))
    logits = torch.tensor([[[1.0, 4.0, 2.0, 3.0, 0.0, -1.0, -2.0, -3.0, -4.0, 9.0]]])

    probs = sampler._effective_log_probs(logits).exp()

    assert torch.count_nonzero(probs > 0).item() == 1
    assert probs.argmax(dim=-1).item() == 1


def test_dream_temperature_and_top_p_filtering():
    sampler = _build_sampler(_config(dream_top_p=0.8, dream_top_k=None, cat_temperature=1.0))
    logits = torch.tensor([[[0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]]])

    log_probs = sampler._effective_log_probs(logits)
    retained = torch.nonzero(log_probs[0, 0].exp() > 0, as_tuple=False).squeeze(-1)

    assert retained.tolist() == [1, 3]

    temperature_sampler = _build_sampler(
        _config(dream_top_p=None, dream_top_k=None, cat_temperature=0.5),
    )
    log_probs = temperature_sampler._effective_log_probs(logits)
    torch.testing.assert_close(log_probs[0, 0, 1] - log_probs[0, 0, 3], torch.tensor(2.0))


def test_dream_internal_scores_use_raw_logits_after_top_p_filtering():
    sampler = _build_sampler(_config(dream_top_p=0.9, dream_top_k=1, cat_temperature=0.5))

    _, scores = sampler.sample("question", return_internal_scores=True)

    # The final-step top-k distribution can assign -inf to tokens committed
    # earlier; reported scores must remain finite raw model log-probabilities.
    assert torch.all(torch.isfinite(scores))
    assert torch.all(scores > -100)


def test_dream_d5p4_selection_expands_each_selected_parent():
    config = _config(
        dream_steps=2,
        gen_length=2,
        n_groups=2,
        group_size=2,
        method="greedy_map",
        subsample_start=0,
        subsample_end=0,
    )
    sampler = _build_sampler(config)
    selector = _RecordingSelector(config)
    sampler.selector = selector

    samples = sampler.sample("question")

    assert samples.shape == (4, 4)
    assert selector.cache_shapes == [
        {
            "x": (4, 2),
            "log_p_x0": (4, 2, 10),
            "embeddings": (4, 2, 3),
        },
    ]


def test_dream_selection_window_is_inclusive_and_step_bounded():
    config = _config(
        dream_steps=3,
        gen_length=2,
        n_groups=2,
        group_size=2,
        method="greedy_map",
        subsample_start=1,
        subsample_end=1,
    )
    sampler = _build_sampler(config)
    selector = _RecordingSelector(config)
    sampler.selector = selector

    sampler.sample("question")

    assert len(selector.cache_shapes) == 1


def test_dream_seeded_sampling_is_reproducible():
    config = _config(dream_alg="origin", cat_temperature=0.7)
    first = _build_sampler(config)
    second = _build_sampler(config)

    torch.manual_seed(17)
    first_samples = first.sample("question")
    torch.manual_seed(17)
    second_samples = second.sample("question")

    assert torch.equal(first_samples, second_samples)


def test_dream_distributed_config_keeps_group_counts_global() -> None:
    global_config = _config(
        n_groups=4,
        group_size=4,
        method="greedy_map",
    )

    local_config = _localize_distributed_config(global_config, world_size=2)

    assert global_config.n_groups == 4
    assert global_config.batch_size == 16
    assert local_config.n_groups == 2
    assert local_config.batch_size == 8

    with pytest.raises(ValueError, match="must be divisible"):
        _localize_distributed_config(global_config, world_size=3)


def test_dream_token_draws_are_invariant_to_two_rank_partition() -> None:
    config = _config(
        seed=23,
        n_groups=4,
        group_size=4,
        method="greedy_map",
        cat_temperature=0.7,
    )
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn((4, 3, 10), generator=generator)
    log_probs = torch.log_softmax(logits, dim=-1)

    single = _build_sampler(config)
    single_samples, _ = single._sample_tokens(log_probs, 4, step=7, sample_index=2)

    rank_samples = []
    for rank in range(2):
        local_config = _config(
            seed=23,
            n_groups=2,
            group_size=4,
            method="greedy_map",
            cat_temperature=0.7,
        )
        local = _build_sampler(local_config)
        local.distributed_utils = SimpleNamespace(rank=rank)
        samples, _ = local._sample_tokens(
            log_probs[rank * 2 : (rank + 1) * 2],
            4,
            step=7,
            sample_index=2,
        )
        rank_samples.append(samples)

    assert torch.equal(single_samples, torch.cat(rank_samples))

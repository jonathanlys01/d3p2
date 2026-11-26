"""A1: post-hoc reranking of independent pools with D5P4's kernel and selection code."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Cache, Config
from d5p4.diffusion_llada import LLADASampler
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample.base import normalized_flat, raw_scores
from d5p4.subsample.beam import DiverseBeamSearch
from d5p4.subsample.greedy_map import GreedyMAP
from d5p4.waves import fronts, posthoc, transfer
from d5p4.waves.spec import Task, d5p4, diverse_beam, value_tag


TOY = Task(name="toy", entry="single_run_llada.py", family="qa", limit_key="qa_dataset_len", n_groups=3, group_size=4)


def _pool(n=12, length=5, dim=8, vocab=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n, length, dim, generator=g)
    # A shared component makes the candidates similar, as LLaDA's hidden states are.
    embeddings = embeddings + 2.0 * torch.randn(1, length, dim, generator=g)
    log_p = F.log_softmax(3.0 * torch.randn(n, length, vocab, generator=g), dim=-1)
    return Cache(embeddings=embeddings, log_p_x0=log_p)


def _features(cache):
    flat = normalized_flat(cache.embeddings)
    return flat @ flat.T, raw_scores(cache)


def _sampler_config(task, method, transversal, **overrides):
    run = d5p4(task, 20.0) if method == "greedy_map" else diverse_beam(task, 20.0)
    return Config(disable_sys_args=True, **{**run.all_overrides(), "transversal": transversal, **overrides})


def test_sources_cover_the_planned_pools():
    sources = posthoc.sources()
    names = [s.name for s in sources]
    assert len(sources) == 31
    assert names[:3] == ["p1/gsm8k/ind_t0p7_s0of4", "p1/gsm8k/ind_t1_s0of4", "p1/gsm8k/ind_t1p3_s0of4"]
    assert [n for n in names if n.startswith("b1/")] == [
        "b1/gsm8k_tempered/ind_t1_tau0p1_seed0_s0of4",
        "b1/gsm8k_tempered/ind_t1_tau0p3_seed0_s0of4",
    ]
    assert sum(n.startswith("b2_full/") for n in names) == 6
    assert sum(n.startswith("scale/") for n in names) == 5
    assert all(s.run.method == "baseline" and s.run.all_overrides()["group_size"] == 1 for s in sources)
    assert {s.run.task.pool_size for s in sources} == {9, 16, 32, 64, 128}


def test_every_source_with_a_search_temperature_has_its_d5p4_counterpart():
    sources = posthoc.sources()
    counterparts = {(s.run.task.name, fronts.base_tag(s.run.tag)) for s in posthoc.counterparts({"gsm8k_tempered"})}
    for source in sources:
        if source.run.task.name != "gsm8k_tempered":
            continue
        base = fronts.base_tag(source.run.tag)
        assert ("gsm8k_tempered", base.replace("ind_", "d5p4_w20_", 1)) in counterparts
        assert ("gsm8k_tempered", base.replace("ind_", "dbs_a80_", 1)) in counterparts


@pytest.mark.parametrize("transversal", [True, False])
def test_posthoc_dpp_matches_the_in_sampler_greedy_map(transversal):
    cache = _pool()
    expected = GreedyMAP(_sampler_config(TOY, "greedy_map", transversal)).subsample(cache)
    gram, neg_entropy = _features(cache)
    picks = posthoc.select_pool(TOY, gram, neg_entropy, betas=(20.0,), alphas=())
    selection = posthoc.DPP_TRANSVERSAL if transversal else posthoc.DPP
    assert picks[(selection, 20.0)] == expected.tolist()


@pytest.mark.parametrize("transversal", [True, False])
def test_posthoc_mmr_matches_diverse_beam_search(transversal):
    cache = _pool(seed=3)
    config = _sampler_config(TOY, "diverse_beam", transversal, _diversity_alpha=20.0)
    expected = DiverseBeamSearch(config).subsample(cache)
    gram, neg_entropy = _features(cache)
    picks = posthoc.select_pool(TOY, gram, neg_entropy, betas=(), alphas=(20.0,))
    selection = posthoc.MMR_TRANSVERSAL if transversal else posthoc.MMR
    assert picks[(selection, 20.0)] == expected.tolist()


def test_gram_factor_reproduces_inner_products():
    gram, _ = _features(_pool())
    factor = posthoc.gram_factor(gram)
    assert torch.allclose(factor @ factor.T, gram, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_transversal_selection_takes_one_per_group(seed):
    gram, neg_entropy = _features(_pool(seed=seed))
    picks = posthoc.select_pool(TOY, gram, neg_entropy)
    for (selection, _), chosen in picks.items():
        assert len(chosen) == len(set(chosen)) == TOY.n_groups
        if selection in (posthoc.DPP_TRANSVERSAL, posthoc.MMR_TRANSVERSAL):
            assert sorted(i // TOY.group_size for i in chosen) == list(range(TOY.n_groups))


@pytest.mark.parametrize("seed", range(5))
def test_vanishing_beta_picks_top_quality(seed):
    gram, neg_entropy = _features(_pool(seed=seed))
    picks = posthoc.select_pool(TOY, gram, neg_entropy, betas=(1e-6,), alphas=())
    per_group = neg_entropy.view(TOY.n_groups, TOY.group_size).argmax(dim=1)
    best_per_group = (per_group + torch.arange(TOY.n_groups) * TOY.group_size).tolist()
    assert sorted(picks[(posthoc.DPP_TRANSVERSAL, 1e-6)]) == sorted(best_per_group)
    assert sorted(picks[(posthoc.DPP, 1e-6)]) == sorted(picks[(posthoc.TOP_Q, None)])
    assert sorted(picks[(posthoc.TOP_Q, None)]) == sorted(torch.topk(neg_entropy, TOY.n_groups).indices.tolist())


def test_large_beta_spreads_the_selection():
    # Two tight clusters: pure quality takes three from the better cluster, a strong kernel term spans both.
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(2, 64, generator=g)
    flat = F.normalize(centers.repeat_interleave(6, dim=0) + 0.01 * torch.randn(12, 64, generator=g), dim=-1)
    neg_entropy = torch.tensor([1.0] * 6 + [0.0] * 6) + 0.01 * torch.rand(12, generator=g)
    picks = posthoc.select_pool(TOY, flat @ flat.T, neg_entropy, betas=(1e-6, 320.0), alphas=())
    assert {i // 6 for i in picks[(posthoc.DPP, 1e-6)]} == {0}
    assert {i // 6 for i in picks[(posthoc.DPP, 320.0)]} == {0, 1}


def _source(name):
    return next(s for s in posthoc.sources() if s.name == name)


def _toy_data(n_questions=2, pool=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    words = ["red", "green", "blue", "cat", "dog", "tree", "sun", "moon", "sea", "sky", "rock", "leaf"]
    texts = [
        [" ".join(words[int(j)] for j in torch.randint(0, len(words), (6,), generator=g)) for _ in range(pool)]
        for _ in range(n_questions)
    ]
    data = {
        "text_samples": texts,
        "references": [[t[0]] for t in texts],
        "internal_scores": torch.randn(n_questions, pool, generator=g).tolist(),
        "dataset_indices": [5 + 4 * i for i in range(n_questions)],
    }
    features = {
        "dataset_index": data["dataset_indices"],
        "gram": torch.stack([_features(_pool(n=pool, seed=seed + i))[0] for i in range(n_questions)]),
        "neg_entropy": torch.stack([_features(_pool(n=pool, seed=seed + i))[1] for i in range(n_questions)]),
    }
    return data, features


def test_posthoc_records_follow_the_fronts_format():
    run = _source("transfer/truthful_qa/ind_t0p7")  # QA family, pool 3 x 3
    data, features = _toy_data()
    records = posthoc.posthoc_records(run.run, data, features)
    assert all(list(r) == fronts.PER_QUESTION_FIELDS for r in records)
    assert {r["question"] for r in records} == {5, 9}
    per_question = len(posthoc.BETAS) * 2 + len(posthoc.ALPHAS) * 2 + 1
    assert len(records) == 2 * per_question
    tags = {(r["selection"], r["tag"]) for r in records}
    assert (posthoc.DPP, "ind_t0p7_w20") in tags
    assert (posthoc.MMR_TRANSVERSAL, "ind_t0p7_a80") in tags
    assert (posthoc.TOP_Q, "ind_t0p7") in tags
    assert all(r["method"] == posthoc.METHOD and 0 <= r["quality"] <= 1 for r in records)
    assert all(r["quality"] >= r["individual"] for r in records)


def test_posthoc_points_pair_with_search_runs_in_summaries():
    source = _source("transfer/truthful_qa/ind_t1")
    data, features = _toy_data(n_questions=3)
    records = fronts.run_question_metrics(source.run, data) + posthoc.posthoc_records(source.run, data, features)
    d5 = d5p4(transfer.TRUTHFUL_QA, 20.0)
    records += [
        {**r, "tag": d5.tag, "method": "greedy_map", "selection": "internal score", "quality": 0.5}
        for r in records
        if r["selection"] == fronts.REFERENCE_SELECTION
    ]
    points, front_rows = fronts.summarize(records, [source.run, d5, *posthoc.pseudo_runs(source.run)], samples=50)
    by_tag = {(p["selection"], p["tag"]): p for p in points}
    assert by_tag[(posthoc.DPP, "ind_t1_w80")]["param_value"] == 80.0
    assert by_tag[(posthoc.DPP, "ind_t1_w80")]["temperature"] == "1"
    assert any(r["selection"] == posthoc.DPP and "hv_minus_d5p4" in r for r in front_rows)

    paired = posthoc.paired_rows(records, posthoc.comparison_pairs([source.run]), samples=50)
    row = next(r for r in paired if r["comparison"] == f"{posthoc.DPP} - D5P4" and r["a_tag"] == "ind_t1_w20")
    assert row["b_tag"] == "d5p4_w20_t1"
    assert row["questions"] == 3
    assert row["quality_b"] == pytest.approx(50.0)
    assert row["quality_diff"] == pytest.approx(row["quality_a"] - 50.0)


def test_comparison_pairs_map_tempered_pools_to_their_search_runs():
    source = next(s for s in posthoc.sources() if s.run.tag == "ind_t1_tau0p3_seed0_s0of4")
    pairs = posthoc.comparison_pairs([source.run])
    b_tags = {b[2] for _, label, _, b in pairs if label.endswith("- D5P4")}
    assert f"d5p4_w{value_tag(20.0)}_t1_tau0p3" in b_tags
    assert "dbs_a80_t1_tau0p3" in {b[2] for _, label, _, b in pairs if label.endswith("- DBS")}


class _StubSampler:
    """Deterministic stand-in for LLADASampler's forward: logits and a final hidden state per token."""

    def __init__(self, vocab=64, dim=6):
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(64, dim, generator=g)
        self.head = torch.randn(dim, vocab, generator=g)
        self.device = "cpu"
        self.config = SimpleNamespace(logits_eos_inf=False, _score_method="entropy")
        self._score_generation_sequences = LLADASampler._score_generation_sequences

    def _forward_model(self, x, *, output_hidden_states=True, logits_slice=None):
        assert output_hidden_states
        hidden = torch.tanh(self.table[x] + self.table[x.roll(1, dims=1)])
        return (hidden @ self.head)[:, logits_slice], (hidden,)


def test_candidate_features_are_chunk_invariant_and_match_the_selector_inputs():
    sampler = _StubSampler()
    g = torch.Generator().manual_seed(1)
    prompt_len, tokens = 4, torch.randint(0, 64, (10, 4 + 7), generator=g)
    whole = posthoc.candidate_features(sampler, tokens, prompt_len, rows_per_forward=16)
    chunked = posthoc.candidate_features(sampler, tokens, prompt_len, rows_per_forward=3)
    for key in whole:
        assert torch.allclose(whole[key].float(), chunked[key].float(), atol=1e-6), key

    logits, (hidden,) = sampler._forward_model(tokens, logits_slice=slice(prompt_len, None))
    cache = Cache(log_p_x0=F.log_softmax(logits, dim=-1), embeddings=hidden[:, prompt_len:])
    gram, neg_entropy = _features(cache)
    assert torch.allclose(whole["gram"], gram, atol=1e-6)
    assert torch.allclose(whole["gram_content"], gram, atol=1e-6)  # no end-of-text token in these rows
    assert torch.allclose(whole["gram"].diagonal(), torch.ones(10), atol=1e-6)
    assert torch.allclose(whole["neg_entropy"], neg_entropy, atol=1e-6)
    assert torch.allclose(-whole["entropy"].float().mean(dim=1), neg_entropy, atol=1e-2)
    assert whole["entropy"].shape == whole["token_logprob"].shape == (10, 7)


def test_parse_indices():
    assert posthoc.parse_indices("all", 3) == [0, 1, 2]
    assert posthoc.parse_indices("0-2,5", 6) == [0, 1, 2, 5]
    with pytest.raises(AssertionError):
        posthoc.parse_indices("4", 3)


def _tiny_d5p4_sampler(gen_length=6):
    torch.manual_seed(0)
    model_config = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=61,
        embedding_size=64,
        max_sequence_length=32,
        mask_token_id=60,
        rope=True,
        alibi=False,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        weight_tying=True,
        init_device="cpu",
    )
    model = LLaDAModelLM(model_config, init_params=True).eval()
    overrides = {
        **transfer.EVAL_SPLIT,
        "gen_length": gen_length,
        "llada_steps": gen_length,
        "block_length": gen_length,
        "subsample_end": gen_length,
        "confidence_eos_eot_inf": False,
    }
    task = Task(name="tiny", entry="single_run_llada.py", family="qa", limit_key="qa_dataset_len", n_groups=2,
                group_size=3, overrides=overrides)
    config = Config(disable_sys_args=True, **d5p4(task, 20.0).all_overrides())

    seen = []

    class Capture(GreedyMAP):
        def subsample(self, cache):
            seen.append(Cache(x=cache.x.clone(), embeddings=cache.embeddings.clone(), log_p_x0=cache.log_p_x0.clone()))
            return super().subsample(cache)

    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.model, sampler.config, sampler.device = model, config, "cpu"
    sampler.mask_index, sampler.sequence_length = 60, 32
    sampler.selector, sampler.distributed_utils = Capture(config), None
    sampler._forward_call_count, sampler.last_forward_count = 0, 0
    prompt = torch.tensor([[5, 9, 17, 3]])
    sampler._preprocess_prompt = MethodType(lambda _self, _prompt: prompt, sampler)
    return sampler, task, prompt, seen


def test_features_equal_what_the_d5p4_selector_sees_at_a_step():
    """In the real sampler loop, re-forwarding a step's states reproduces the selector's kernel inputs exactly."""
    sampler, task, prompt, seen = _tiny_d5p4_sampler()
    sampler.sample("unused")
    assert len(seen) == 6
    for cache in (seen[1], seen[-1]):  # an early step and the last selection (one position still masked)
        tokens = torch.cat([prompt.expand(cache.x.size(0), -1), cache.x], dim=1)
        features = posthoc.candidate_features(sampler, tokens, prompt.size(1), rows_per_forward=task.pool_size)
        gram, neg_entropy = _features(cache)
        torch.testing.assert_close(features["gram"], gram)
        torch.testing.assert_close(features["neg_entropy"], neg_entropy)
        # And the selection from the stored features is the one D5P4 made at that step.
        picks = posthoc.select_pool(task, features["gram"], features["neg_entropy"], betas=(20.0,), alphas=())
        assert picks[(posthoc.DPP_TRANSVERSAL, 20.0)] == GreedyMAP(sampler.config).subsample(cache).tolist()

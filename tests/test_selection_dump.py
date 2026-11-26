"""Selection dumps (A3) and their offline exact / greedy MAP analysis."""

import json
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from d5p4 import selection_dump_analysis as analysis
from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler
from d5p4.resume_db import semantic_config_dict
from d5p4.selection_dump import SelectionDumpRecorder, dump_path, load_dump
from d5p4.subsample.baseline import BaselineSelection
from d5p4.subsample.greedy_map import GreedyMAP, _greedy_map_cpu


MASK_TOKEN_ID = 15
VOCAB_SIZE = 16
HIDDEN_SIZE = 6
PROMPT = torch.tensor([[1, 2, 3]], dtype=torch.long)


def _random_kernel(n: int, rng: np.random.Generator, beta: float = 20.0, cos: float = 0.9) -> np.ndarray:
    x = analysis.unit_vectors_with_mean_cosine(n, cos, 64, rng)
    q = rng.random(n)
    return beta * (x @ x.T) + np.diag(q)


# ── Enumeration and greedy ────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("k", "w", "count"), [(3, 3, 27), (4, 4, 256), (8, 4, 65_536)])
def test_transversal_sets_enumerate_every_one_per_group_choice(k, w, count):
    sets = analysis.transversal_sets(k, w)
    assert sets.shape == (count, k)
    assert np.array_equal(sets // w, np.broadcast_to(np.arange(k), sets.shape))
    assert len({tuple(s) for s in sets}) == count


def test_exact_map_matches_brute_force_and_bounds_greedy():
    rng = np.random.default_rng(0)
    k, w = 3, 3
    for _ in range(20):
        L = _random_kernel(k * w, rng)
        brute = max(
            (np.linalg.slogdet(L[np.ix_(s, s)])[1], s)
            for s in (list(c) for c in product(*[range(g * w, g * w + w) for g in range(k)]))
        )
        result = analysis.solve_step(L, k, w)
        assert result["opt"] == pytest.approx(brute[0], abs=1e-9)
        assert np.array_equal(result["opt_set"], brute[1])
        assert result["single"] <= result["multi"] + 1e-12 <= result["opt"] + 2e-12


def test_multi_start_greedy_matches_the_production_cpu_solver():
    rng = np.random.default_rng(1)
    for k, w in ((3, 3), (4, 4), (8, 4)):
        groups = np.repeat(np.arange(k), w)
        item_to_group = torch.from_numpy(groups)
        members = torch.arange(k * w).view(k, w)
        for _ in range(10):
            L = _random_kernel(k * w, rng, cos=0.5)
            production = _greedy_map_cpu(
                torch.from_numpy(L),
                k,
                item_to_group,
                members,
                torch.arange(k * w),
            ).numpy()
            mine, _ = analysis.best_of(L, analysis.greedy_transversal(L, groups, np.arange(k * w)))
            assert np.array_equal(np.sort(production), mine)


def test_single_start_greedy_is_the_trajectory_from_the_largest_diagonal():
    rng = np.random.default_rng(2)
    L = _random_kernel(16, rng)
    groups = np.repeat(np.arange(4), 4)
    single = analysis.greedy_transversal(L, groups)
    every = analysis.greedy_transversal(L, groups, np.arange(16))
    assert np.array_equal(single[0], every[int(np.argmax(np.diag(L)))])


def test_gap_metrics_on_a_known_optimum():
    result = {"opt": 1.0, "mean": 0.0, "opt_set": np.array([0, 2]), "values": np.array([1.0, 0.5, -0.5, -1.0])}
    row = analysis.gap_row(result, 0.5, np.array([0, 3]), "g")
    assert row["g_delta"] == pytest.approx(0.5)
    assert row["g_det_ratio"] == pytest.approx(np.exp(-0.5))
    assert row["g_norm_gap"] == pytest.approx(0.5)
    assert row["g_optimal"] == 0
    assert row["g_rank"] == pytest.approx(0.25)


def test_fnw_bound_holds_for_plain_greedy():
    rng = np.random.default_rng(3)
    for _ in range(50):
        L = _random_kernel(16, rng, beta=1.0)
        result = analysis.solve_step(L, 4, 4)
        assert result["single"] >= analysis.fnw_bound(L, result["opt"], 4) - 1e-9


def test_ancestors_follow_selected_rows_back_through_groups():
    # w = 2, k = 2. Step 0 keeps rows 1 and 2; step 1 is outside the window; step 2 keeps rows 3 and 0.
    sel = np.array([[1, 2], [-1, -1], [3, 0]], dtype=np.int16)
    selected = np.array([True, False, True])
    anc = analysis.ancestors(sel, selected, 2, np.array([0, 3]))
    # Final row 0 is in group 0, the child of row 3 at step 2 -> row 3 at step 1 -> group 1's parent at step 0: row 2.
    assert anc.tolist() == [[2, 1], [3, 0], [3, 0]]


# ── The dump flag inside the sampler ─────────────────────────────────────────────────────────────────────────────


class TinyLM(nn.Module):
    """Small deterministic stand-in whose logits and hidden states depend on every token."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=g)
        self.pos = 0.3 * torch.randn(64, HIDDEN_SIZE, generator=g)
        self.out = torch.randn(HIDDEN_SIZE, VOCAB_SIZE, generator=g)
        self.config = SimpleNamespace(mask_token_id=MASK_TOKEN_ID, vocab_size=VOCAB_SIZE)

    def forward(self, input_ids, *, return_dict, output_hidden_states, last_hidden_state_only, logits_slice):
        assert return_dict and last_hidden_state_only
        hidden = self.emb[input_ids] + self.pos[: input_ids.shape[1]]
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        logits = (hidden @ self.out).float()
        logits[..., MASK_TOKEN_ID] = -1e4
        if logits_slice is not None:
            logits = logits[:, logits_slice]
        return CausalLMOutputWithPast(logits=logits, hidden_states=(hidden,) if output_hidden_states else None)


def _config(tmp_path, **overrides) -> Config:
    base = {
        "disable_sys_args": True,
        "model": "llada",
        "cfg_scale": 0.0,
        "llada_reference_cfg": True,
        "cat_temperature": 1.0,
        "confidence_eos_eot_inf": False,
        "gen_length": 8,
        "block_length": 8,
        "llada_steps": 8,
        "subsample_start": 0,
        "subsample_end": 8,
        "n_groups": 3,
        "group_size": 3,
        "transversal": True,
        "method": "greedy_map",
        "_kernel_type": "cosine",
        "_kernel_method": "additive",
        "_w_interaction": 20.0,
        "results_dir": str(tmp_path),
    }
    base.update(overrides)
    return Config(**base)


def _sampler(config: Config) -> LLADASampler:
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.model = TinyLM()
    sampler.selector = GreedyMAP(config)
    sampler.selector.device = "cpu"
    sampler.distributed_utils = None
    sampler.mask_index = MASK_TOKEN_ID
    sampler._forward_call_count = 0
    sampler.last_forward_count = 0
    sampler._preprocess_prompt = lambda _prompt: PROMPT.clone()
    return sampler


def _run(config: Config, prompt: str = "question"):
    torch.manual_seed(0)
    return _sampler(config).sample(prompt, return_internal_scores=True)


def test_dump_is_off_by_default_and_creates_nothing(tmp_path):
    assert Config(disable_sys_args=True).selection_dump is False
    config = _config(tmp_path)
    assert SelectionDumpRecorder.create(config, "q", 3, None, distributed=False) is None
    _run(config)
    assert not (tmp_path / "selection_dumps").exists()


def test_dump_does_not_change_generation_and_records_the_solver_inputs(tmp_path):
    x_off, scores_off = _run(_config(tmp_path))
    config = _config(tmp_path, selection_dump=True)
    x_on, scores_on = _run(config)
    assert torch.equal(x_off, x_on)
    assert torch.equal(scores_off, scores_on)

    dump = load_dump(dump_path(config, "question"))
    steps, pool, k = config.llada_steps, config.batch_size, config.n_groups
    assert dump["sel"].shape == (steps, k) and dump["selected"].all()
    assert dump["L"].shape == (steps, pool, pool) and dump["S"].shape == (steps, pool, pool)
    assert dump["L"].dtype == np.float32
    assert dump["kernel_steps"].tolist() == list(range(steps))
    assert dump["score_neg_entropy"].shape == (steps, pool)
    assert dump["mask_frac"][0] == 1.0 and np.all(np.diff(dump["mask_frac"]) < 0)
    assert np.array_equal(dump["final_tokens"], x_on[:, PROMPT.shape[1] :].numpy())
    assert np.array_equal(dump["final_internal_scores"], np.asarray(scores_on.tolist()))
    assert dump["meta"]["w_interaction"] == 20.0 and dump["prompt"] == "question"

    # The dumped kernel is the production solver's input: rerunning the solver on it gives the dumped selection.
    groups = torch.arange(k).repeat_interleave(config.group_size)
    members = torch.arange(pool).view(k, config.group_size)
    for L, sel in zip(dump["L"], dump["sel"], strict=True):
        again = _greedy_map_cpu(torch.from_numpy(L), k, groups, members, torch.arange(pool))
        assert sorted(again.tolist()) == sorted(sel.tolist())
    # Additive kernel: off-diagonal entries are beta * cosine, the diagonal adds the normalized quality in [0, 1].
    off = ~np.eye(pool, dtype=bool)
    assert np.allclose(dump["L"][:, off], 20.0 * dump["S"][:, off], atol=1e-3)
    q = np.diagonal(dump["L"], axis1=1, axis2=2) - 20.0 * np.diagonal(dump["S"], axis1=1, axis2=2)
    assert q.min() > -1e-3 and q.max() < 1 + 1e-3


def test_dump_every_thins_kernels_but_keeps_ancestry(tmp_path):
    config = _config(tmp_path, selection_dump=True, selection_dump_every=3)
    _run(config)
    dump = load_dump(dump_path(config, "question"))
    assert dump["kernel_steps"].tolist() == [0, 3, 6]
    assert dump["sel"].shape[0] == config.llada_steps


def test_independent_sampling_dumps_scores_and_identity_ancestry(tmp_path):
    config = _config(tmp_path, selection_dump=True, method="baseline", n_groups=4, group_size=1)
    sampler = _sampler(config)
    sampler.selector = BaselineSelection(config)
    sampler.selector.device = "cpu"
    torch.manual_seed(0)
    sampler.sample("question", return_internal_scores=True)
    dump = load_dump(dump_path(config, "question"))
    assert dump["L"].shape[0] == 0
    assert np.array_equal(dump["sel"], np.tile(np.arange(4), (config.llada_steps, 1)))
    assert dump["score_neg_entropy"].shape == (config.llada_steps, 4)


def test_dump_flags_do_not_change_the_resume_hash(tmp_path):
    off = semantic_config_dict(_config(tmp_path))
    on = semantic_config_dict(_config(tmp_path, selection_dump=True, selection_dump_every=4))
    assert off == on
    assert "selection_dump" not in off


def test_dump_requires_the_diffusion_decoder():
    with pytest.raises(AssertionError, match="selection_dump"):
        Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            method="ltr_beam",
            cfg_scale=1.0,
            n_groups=2,
            group_size=2,
            selection_dump=True,
        )


def test_solver_and_proxy_commands_run_on_dumps(tmp_path):
    results = {"text_samples": [], "internal_scores": [], "results": []}
    for index in range(3):
        config = _config(tmp_path, selection_dump=True)
        x, scores = _run(config, prompt=f"question {index}")
        results["text_samples"].append(["text"] * config.batch_size)
        results["internal_scores"].append(scores.tolist())
        results["results"].append({"scores": [float(i % 2) for i in range(config.batch_size)]})
    results_path = tmp_path / "math-run.json"
    results_path.write_text(json.dumps(results))
    dump_dir = tmp_path / "selection_dumps"

    assert analysis.main(["solver", str(dump_dir), "--out", str(tmp_path / "solver")]) == 0
    summary = json.loads((tmp_path / "solver" / "summary.json").read_text())
    assert summary["steps"] == 3 * 8
    assert summary["degenerate_steps"] >= 3  # step 0: every candidate is the same fully masked state
    if summary["informative_steps"]:
        assert summary["multi"]["optimal_frac"] >= summary["single"]["optimal_frac"]
        assert summary["prod_eq_multi_frac"] >= 0.99

    assert (
        analysis.main(["proxy", str(dump_dir), "--results", str(results_path), "--out", str(tmp_path / "proxy")]) == 0
    )
    proxy = json.loads((tmp_path / "proxy" / "summary.json").read_text())
    assert {row["mask_frac"] for row in proxy["neg_entropy"]} >= {"final"}

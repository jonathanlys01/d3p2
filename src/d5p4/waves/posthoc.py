"""A1: post-hoc DPP reranking of independent-sampling pools, the control for "selecting during denoising matters".

Independent sampling (IND) generates the same pool as D5P4, N = k * w candidates per question, with no selection.
This module reranks those pools after generation with D5P4's own kernel and selection code, so the only difference
from D5P4 is when the selection happens (once, on finished texts, instead of at every denoising step):

1. `features` (GPU, one Slurm array task per IND run): reads each question's N token rows (prompt + completion) from
   the run's resume database, runs one LLaDA forward over each fully unmasked row exactly as the sampler does at a
   selection step (`LLADASampler._forward_model`, bf16 autocast, final hidden state after the last layer norm, logits
   over the generation positions), and stores per question only the N x N cosine Gram matrix of the flattened,
   normalized hidden states (`normalized_flat`, as in `BaseSelector.compute_kernel`) and the raw quality score of
   each candidate (`raw_scores`, mean negative token entropy over the generation positions, before the pool-wise
   min-max normalization). Kept as extras for robustness checks: per-position entropies and token log probs, and a
   Gram matrix over the answer tokens only (end-of-text padding zeroed).
2. `select` (CPU, offline): per question, normalizes the scores over the pool (`normalize_scores`), builds the kernel
   for each beta with `combine_kernel` and selects k of N with the production greedy MAP (`GreedyMAP._transversal` /
   `_non_transversal`): without a partition ("post-hoc DPP") and with the pool split into its k consecutive blocks of
   w ("post-hoc DPP transversal"; IND samples are exchangeable, so this is a uniformly random partition). MMR uses
   Diverse Beam Search's `_diverse_beam_full_explore` on a factor of the Gram matrix (it only reads inner products).
   Records follow `fronts.PER_QUESTION_FIELDS` with method "posthoc", so they pair by question with the D5P4 and DBS
   runs of the same task; the IND run's own "random k of N" and "internal score" points come from `fronts`.

Outputs live in their own root (default `$WAVES_ROOT/a1_posthoc`): `features/<wave>/<task>/<tag>/features.pt`, then
`per_question.csv`, `points.csv`, `fronts.csv` and `paired.csv` from `select`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from d5p4.config import Cache, Config
from d5p4.eval_utils import _compute_self_bleu_impl
from d5p4.subsample.base import combine_kernel, normalize_scores, normalized_flat, raw_scores
from d5p4.subsample.beam import _diverse_beam_full_explore
from d5p4.subsample.greedy_map import GreedyMAP
from d5p4.waves import fronts, get_wave
from d5p4.waves.spec import FRONT_DIVERSITY_ALPHAS, FRONT_W_INTERACTIONS, Run, Task, d5p4, value_tag


BETAS = FRONT_W_INTERACTIONS  # 5, 20, 80, 320: the frozen D5P4 grid
ALPHAS = FRONT_DIVERSITY_ALPHAS  # 5, 20, 80, 320: the frozen DBS grid
METHOD = "posthoc"
DPP = "post-hoc DPP"
DPP_TRANSVERSAL = "post-hoc DPP transversal"
MMR = "post-hoc MMR"
MMR_TRANSVERSAL = "post-hoc MMR transversal"
TOP_Q = "post-hoc top Q"
FEATURES_FILE = "features.pt"
ROWS_PER_FORWARD = 16
# <|endoftext|> (the padding after an answer) and <|eot_id|>: masked out of the content-only Gram matrix.
LLADA_END_IDS = (126081, 126348)

# IND pools to rerank, by wave. b1: only the tempered GSM8K pools that B2's search runs pair with (seed 0, shard 0).
SOURCE_WAVES = ("p1", "transfer", "b1", "b2_full", "scale")
B1_TAGS = ("ind_t1_tau0p1_seed0_s0of4", "ind_t1_tau0p3_seed0_s0of4")
# Waves whose D5P4 and DBS runs are the paired counterparts (b2 holds the tempered search runs on shard 0).
COUNTERPART_WAVES = ("p1", "transfer", "b2", "b2_full", "scale")
SEARCH_METHODS = ("greedy_map", "diverse_beam")


@dataclass(frozen=True)
class Source:
    wave: str
    run: Run

    @property
    def name(self) -> str:
        return f"{self.wave}/{self.run.task.name}/{self.run.tag}"

    def wave_root(self, waves_root: Path) -> Path:
        return waves_root / self.wave

    def features_path(self, out_root: Path) -> Path:
        return out_root / "features" / self.wave / self.run.task.name / self.run.tag / FEATURES_FILE


def sources() -> list[Source]:
    """Every IND run A1 reranks, in a fixed order: the index of the `features` Slurm array."""
    result = []
    for wave in SOURCE_WAVES:
        for run in get_wave(wave):
            if run.method != "baseline":
                continue
            if wave == "b1" and (run.task.name != "gsm8k_tempered" or run.tag not in B1_TAGS):
                continue
            result.append(Source(wave, run))
    return result


def counterparts(tasks: Iterable[str]) -> list[Source]:
    """The D5P4 and DBS runs on the reranked tasks, to pair with."""
    wanted = set(tasks)
    return [
        Source(wave, run)
        for wave in COUNTERPART_WAVES
        for run in get_wave(wave)
        if run.method in SEARCH_METHODS and run.task.name in wanted
    ]


# ---------------------------------------------------------------------------------------------------------------------
# GPU: per-question Gram matrices and quality scores
# ---------------------------------------------------------------------------------------------------------------------


def resume_database(source: Source, waves_root: Path, result_path: str) -> Path:
    """The resume database that produced `result_path` (its `runs.result_path`), or the only one there is."""
    candidates = sorted(source.run.resume_dir(source.wave_root(waves_root)).glob("*.sqlite3"))
    matching = []
    for path in candidates:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
            row = con.execute("SELECT result_path FROM runs").fetchone()
        if row is not None and row[0] and os.path.basename(row[0]) == os.path.basename(result_path):
            matching.append(path)
    if len(matching) == 1:
        return matching[0]
    if not matching and len(candidates) == 1:
        print(f"Warning: {source.name}: the resume database does not name {result_path}; using the only one.")
        return candidates[0]
    raise FileNotFoundError(f"{source.name}: {len(matching)} of {len(candidates)} resume databases match {result_path}")


def stored_generations(db_path: Path) -> list[dict[str, Any]]:
    """Token rows (prompt + completion, [N, prompt + gen_length]) and decoded texts per item, in item order."""
    from d5p4.resume_db import tensor_from_blob  # noqa: PLC0415

    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as con:
        rows = con.execute(
            "SELECT item_index, token_ids_blob, prompt_len, internal_scores_json, decoded_json "
            "FROM generations ORDER BY item_index",
        ).fetchall()
    return [
        {
            "item_index": int(item_index),
            "tokens": tensor_from_blob(blob),
            "prompt_len": int(prompt_len),
            "internal_scores": json.loads(scores) if scores else None,
            "decoded": json.loads(decoded) if decoded else None,
        }
        for item_index, blob, prompt_len, scores, decoded in rows
    ]


@torch.no_grad()
def candidate_features(sampler: Any, tokens: torch.Tensor, prompt_len: int, rows_per_forward: int) -> dict:
    """Gram matrix and quality scores of one question's pool, as a D5P4 selection step would compute them.

    The selection step of `LLADASampler.sample` (one block, no forced order) forwards the whole sequence, takes the
    final hidden state and the log-softmax of the logits over the generation positions, and hands them to the
    selector; `compute_kernel` flattens and normalizes the hidden states and scores each candidate by `raw_scores`.
    Rows are independent, so the pool is forwarded in chunks of `rows_per_forward`.
    """
    config = sampler.config
    flats, content_flats, neg_entropy, mean_logprob, entropy, token_logprob = [], [], [], [], [], []
    for start in range(0, tokens.size(0), rows_per_forward):
        x = tokens[start : start + rows_per_forward].to(sampler.device)
        logits, hidden = sampler._forward_model(x, output_hidden_states=True, logits_slice=slice(prompt_len, None))
        if config.logits_eos_inf and logits.size(-1) > 126081:
            logits[:, :, 126081] = -torch.inf
        log_p_x0 = F.log_softmax(logits, dim=-1)
        del logits
        assert hidden is not None
        generation = x[:, prompt_len:]
        embeddings = hidden[-1][:, prompt_len:]
        flats.append(normalized_flat(embeddings))
        content = ~torch.isin(generation, torch.tensor(LLADA_END_IDS, device=generation.device))
        content_flats.append(normalized_flat(embeddings * content.unsqueeze(-1)))
        neg_entropy.append(raw_scores(Cache(log_p_x0=log_p_x0, x=generation), config._score_method))
        mean_logprob.append(sampler._score_generation_sequences(log_p_x0, generation))
        log_p = log_p_x0.float()
        entropy.append(-(log_p.exp() * log_p).sum(dim=-1).half())
        token_logprob.append(torch.gather(log_p, -1, generation.unsqueeze(-1)).squeeze(-1).half())
        del hidden, embeddings, log_p_x0, log_p
    flat, content_flat = torch.cat(flats), torch.cat(content_flats)
    return {
        "gram": torch.matmul(flat, flat.T).cpu(),
        # Extra, not D5P4's kernel: the same cosine over the answer tokens only, without the end-of-text padding.
        "gram_content": torch.matmul(content_flat, content_flat.T).cpu(),
        "neg_entropy": torch.cat(neg_entropy).float().cpu(),
        "mean_logprob": torch.cat(mean_logprob).float().cpu(),
        "entropy": torch.cat(entropy).cpu(),
        "token_logprob": torch.cat(token_logprob).cpu(),
    }


def extract_features(  # noqa: PLR0913
    source: Source,
    sampler: Any,
    waves_root: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    rows_per_forward: int = ROWS_PER_FORWARD,
) -> None:
    run = source.run
    result_path = fronts._result_file(str(run.results_dir(source.wave_root(waves_root))))
    if result_path is None:
        raise FileNotFoundError(f"{source.name}: no result JSON")
    with open(result_path) as f:
        data = json.load(f)
    db_path = resume_database(source, waves_root, result_path)
    generations = stored_generations(db_path)
    texts = data["text_samples"]
    dataset_indices = data.get("dataset_indices") or list(range(len(texts)))
    assert len(generations) == len(texts), f"{source.name}: {len(generations)} stored items, {len(texts)} in JSON"
    if limit is not None:
        generations = generations[:limit]
    print(f"{source.name}: {len(generations)} questions, pool {run.task.pool_size}, db {db_path.name}", flush=True)

    per_question: dict[str, list[torch.Tensor]] = {}
    max_internal_gap = 0.0
    started = time.time()
    for i, generation in enumerate(generations):
        assert generation["item_index"] == i, f"{source.name}: item {generation['item_index']} at position {i}"
        assert generation["decoded"] == texts[i], f"{source.name}: stored texts differ from the JSON at item {i}"
        tokens, prompt_len = generation["tokens"], generation["prompt_len"]
        assert tokens.shape == (run.task.pool_size, prompt_len + run.all_overrides()["gen_length"]), tokens.shape
        features = candidate_features(sampler, tokens, prompt_len, rows_per_forward)
        if generation["internal_scores"] is not None:
            # The stored internal score is the mean token log prob at the last step (one position still masked).
            stored = torch.tensor(generation["internal_scores"], dtype=torch.float32)
            max_internal_gap = max(max_internal_gap, float((features["mean_logprob"] - stored).abs().max()))
        for key, value in features.items():
            per_question.setdefault(key, []).append(value)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(generations)} questions, {time.time() - started:.0f} s", flush=True)

    payload = {
        "wave": source.wave,
        "task": run.task.name,
        "tag": run.tag,
        "result_file": os.path.basename(result_path),
        "resume_db": db_path.name,
        "n_groups": run.task.n_groups,
        "group_size": run.task.group_size,
        "score_method": sampler.config._score_method,
        "dataset_index": [int(dataset_indices[i]) for i in range(len(generations))],
        "gram": torch.stack(per_question["gram"]),
        "gram_content": torch.stack(per_question["gram_content"]),
        "neg_entropy": torch.stack(per_question["neg_entropy"]),
        "mean_logprob": torch.stack(per_question["mean_logprob"]),
        # Per position, float16: the generation length is fixed per task, so these stack.
        "entropy": torch.stack(per_question["entropy"]),
        "token_logprob": torch.stack(per_question["token_logprob"]),
        "max_internal_score_gap": max_internal_gap,
        "wall_time_s": time.time() - started,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial")
    torch.save(payload, partial)
    partial.replace(out_path)
    print(
        f"{source.name}: wrote {out_path} in {payload['wall_time_s']:.0f} s; max |recomputed - stored internal score| "
        f"= {max_internal_gap:.4f}",
        flush=True,
    )


def cmd_features(args: argparse.Namespace) -> int:
    from d5p4.diffusion_llada import LLADASampler  # noqa: PLC0415
    from d5p4.waves.__main__ import build_config  # noqa: PLC0415

    all_sources = sources()
    sampler = None
    for index in parse_indices(args.index, len(all_sources)):
        source = all_sources[index]
        out_path = source.features_path(args.out)
        if out_path.exists() and not args.force:
            print(f"{index}: {source.name} already done ({out_path})")
            continue
        overrides = source.run.all_overrides()
        if args.cache_dir is not None:
            overrides["cache_dir"] = str(args.cache_dir)
        config = build_config(source.run, args.config, overrides)
        if sampler is None:
            sampler = LLADASampler(config)  # eager: generation compiled the model, the forward is the same
        else:
            sampler.update_config(config)
        print(f"{index}: {source.name}", flush=True)
        extract_features(
            source,
            sampler,
            args.waves_root,
            out_path,
            limit=args.limit,
            rows_per_forward=args.rows_per_forward,
        )
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CPU: selection from the stored features
# ---------------------------------------------------------------------------------------------------------------------


class _GivenKernel(GreedyMAP):
    """D5P4's greedy MAP selector applied to a kernel built elsewhere (the stored Gram matrix and scores)."""

    kernel: torch.Tensor

    def compute_kernel(self, cache: Cache) -> torch.Tensor:  # noqa: ARG002
        return self.kernel


_CONFIGS: dict[tuple[str, int, int, float, bool], Config] = {}
_SELECTORS: dict[int, _GivenKernel] = {}


def selection_config(task: Task, beta: float, transversal: bool) -> Config:
    """The configuration of the D5P4 run at `beta` on `task`, with or without the transversal constraint."""
    key = (task.name, task.n_groups, task.group_size, float(beta), transversal)
    if key not in _CONFIGS:
        overrides = {**d5p4(task, beta).all_overrides(), "transversal": transversal}
        _CONFIGS[key] = Config(disable_sys_args=True, **overrides)  # pyrefly: ignore [bad-argument-type]
    return _CONFIGS[key]


def dpp_map(kernel: torch.Tensor, config: Config) -> list[int]:
    """Greedy MAP (multi-start) of the D5P4 selector on `kernel`: k of N, one per group when transversal."""
    if id(config) not in _SELECTORS:
        _SELECTORS[id(config)] = _GivenKernel(config)
    selector = _SELECTORS[id(config)]
    assert selector.config is config
    selector.kernel = kernel
    selected = selector._transversal(Cache()) if config.transversal else selector._non_transversal(Cache())
    assert selected is not None and selector._validate_global_selection(selected.long()), selected
    return [int(i) for i in selected]


def gram_factor(gram: torch.Tensor) -> torch.Tensor:
    """V with V @ V.T = gram (clipping round-off negative eigenvalues): MMR only reads inner products."""
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    return (eigenvectors * eigenvalues.clamp(min=0).sqrt()).float()


def mmr(scores: torch.Tensor, factor: torch.Tensor, k: int, alpha: float, group_size: int | None) -> list[int]:
    """Diverse Beam Search's MMR (Alg. 2, mean similarity, multi-start); `group_size` set = one per group."""
    n = scores.size(0)
    if group_size is None:
        item_to_group = torch.arange(n)
    else:
        item_to_group = torch.arange(n // group_size).repeat_interleave(group_size)
    return [int(i) for i in _diverse_beam_full_explore(scores, factor, k, float(alpha), item_to_group)]


def select_pool(  # noqa: PLR0913
    task: Task,
    gram: torch.Tensor,
    neg_entropy: torch.Tensor,
    *,
    betas: Iterable[float] = BETAS,
    alphas: Iterable[float] = ALPHAS,
) -> dict[tuple[str, float | None], list[int]]:
    """Every post-hoc selection of k from one question's pool of N = k * w, keyed by (selection, parameter)."""
    k, w = task.n_groups, task.group_size
    assert gram.shape == (k * w, k * w), f"Expected a pool of {k} x {w}, got {tuple(gram.shape)}"
    gram = gram.float()
    scores = normalize_scores(neg_entropy.float())
    picks: dict[tuple[str, float | None], list[int]] = {
        (TOP_Q, None): [int(i) for i in torch.argsort(-scores, stable=True)[:k]],
    }
    for beta in betas:
        for name, transversal in ((DPP, False), (DPP_TRANSVERSAL, True)):
            config = selection_config(task, beta, transversal)
            picks[(name, beta)] = dpp_map(combine_kernel(gram.clone(), scores, config), config)
    factor = gram_factor(gram)
    for alpha in alphas:
        picks[(MMR, alpha)] = mmr(scores, factor, k, alpha, None)
        picks[(MMR_TRANSVERSAL, alpha)] = mmr(scores, factor, k, alpha, w)
    return picks


def posthoc_tag(base: str, selection: str, param: float | None) -> str:
    """`ind_t1` -> `ind_t1_w20` (DPP at beta 20), `ind_t1_a80` (MMR at alpha 80), `ind_t1` (top Q)."""
    if param is None:
        return base
    return f"{base}_{'a' if selection in (MMR, MMR_TRANSVERSAL) else 'w'}{value_tag(param)}"


def posthoc_records(
    run: Run,
    data: dict[str, Any],
    features: dict[str, Any],
    gram_key: str = "gram",
) -> list[dict[str, Any]]:
    """Per-question records of every post-hoc selection on one IND run, in `fronts.PER_QUESTION_FIELDS`.

    `gram_key` "gram" is D5P4's kernel; "gram_content" the robustness variant over the answer tokens only.
    """
    base = fronts.base_tag(run.tag)
    records = []
    rows = list(fronts._question_rows(run, data))
    n = len(features["dataset_index"])
    for i, (index, cands, quality, _internal) in enumerate(rows[:n]):
        assert int(index) == features["dataset_index"][i], f"{run.tag}: question {index} vs features at {i}"
        q = np.asarray(quality, dtype=float)
        self_bleu_cache: dict[tuple[int, ...], float] = {}
        pool = select_pool(run.task, features[gram_key][i], features["neg_entropy"][i])
        for (selection, param), picks in pool.items():
            key = tuple(sorted(picks))
            if key not in self_bleu_cache:
                self_bleu_cache[key] = _compute_self_bleu_impl([cands[j] for j in key])
            records.append(
                {
                    "task": run.task.name,
                    "tag": posthoc_tag(base, selection, param),
                    "method": METHOD,
                    "selection": selection,
                    "question": index,
                    "quality": float(q[list(key)].max()),
                    "individual": float(q[list(key)].mean()),
                    "self_bleu": self_bleu_cache[key],
                },
            )
    return records


def pseudo_runs(run: Run) -> list[Run]:
    """One Run per post-hoc point of `run`, so `fronts.summarize` knows its parameter and temperature."""
    base = fronts.base_tag(run.tag)
    temperature = run.all_overrides()["cat_temperature"]
    result = [Run(run.task, base, METHOD, {"cat_temperature": temperature})]
    result += [
        Run(run.task, f"{base}_w{value_tag(b)}", METHOD, {"cat_temperature": temperature, "_w_interaction": b})
        for b in BETAS
    ]
    result += [
        Run(run.task, f"{base}_a{value_tag(a)}", METHOD, {"cat_temperature": temperature, "_diversity_alpha": a})
        for a in ALPHAS
    ]
    return result


def _load_json(source: Source, waves_root: Path) -> dict[str, Any] | None:
    path = fronts._result_file(str(source.run.results_dir(source.wave_root(waves_root))))
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def _source_records(job: tuple[Source, Path, Path, str]) -> list[dict[str, Any]]:
    """The post-hoc records of one IND run, plus its own "random k of N" and "internal score" records."""
    source, waves_root, features_root, gram_key = job
    data = _load_json(source, waves_root)
    features_path = source.features_path(features_root)
    if data is None or not features_path.exists():
        print(f"Skipping {source.name}: {'no result JSON' if data is None else 'no features'}")
        return []
    features = torch.load(features_path, map_location="cpu", weights_only=False)
    covered = set(features["dataset_index"])
    own = [r for r in fronts.run_question_metrics(source.run, data) if int(r["question"]) in covered]
    records = own + posthoc_records(source.run, data, features, gram_key)
    print(f"{source.name}: {len(covered)} questions, {len(records)} records", flush=True)
    return records


def _counterpart_records(job: tuple[Source, Path]) -> list[dict[str, Any]]:
    source, waves_root = job
    data = _load_json(source, waves_root)
    return [] if data is None else fronts.run_question_metrics(source.run, data)


# ---------------------------------------------------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------------------------------------------------


Point = tuple[str, str, str]  # (method, selection, tag)


def comparison_pairs(source_runs: Iterable[Run]) -> list[tuple[str, str, Point, Point]]:
    """(task, label, a, b): post-hoc against D5P4 / DBS at the same parameter and temperature, and against IND."""
    pairs = []
    for run in source_runs:
        task, base = run.task.name, fronts.base_tag(run.tag)
        random_k = ("baseline", fronts.REFERENCE_SELECTION, base)
        pairs.append((task, "top Q - random k of N", (METHOD, TOP_Q, base), random_k))
        for beta in BETAS:
            tag = posthoc_tag(base, DPP, beta)
            search = ("greedy_map", "internal score", base.replace("ind_", f"d5p4_w{value_tag(beta)}_", 1))
            for selection in (DPP, DPP_TRANSVERSAL):
                pairs.append((task, f"{selection} - D5P4", (METHOD, selection, tag), search))
            pairs.append((task, f"{DPP} - {DPP_TRANSVERSAL}", (METHOD, DPP, tag), (METHOD, DPP_TRANSVERSAL, tag)))
            pairs.append((task, f"{DPP} - random k of N", (METHOD, DPP, tag), random_k))
            pairs.append((task, "D5P4 - random k of N", search, random_k))
        for alpha in ALPHAS:
            tag = posthoc_tag(base, MMR, alpha)
            search = ("diverse_beam", "internal score", base.replace("ind_", f"dbs_a{value_tag(alpha)}_", 1))
            for selection in (MMR, MMR_TRANSVERSAL):
                pairs.append((task, f"{selection} - DBS", (METHOD, selection, tag), search))
    return list(dict.fromkeys(pairs))


def paired_rows(
    records: list[dict[str, Any]],
    pairs: list[tuple[str, str, Point, Point]],
    *,
    samples: int = fronts.BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Paired bootstrap over the questions both points answered: differences in quality, individual and Self-BLEU."""
    values: dict[tuple[str, Point], dict[int, list[tuple[float, float, float]]]] = {}
    for r in records:
        key = (r["task"], (r["method"], r["selection"], r["tag"]))
        values.setdefault(key, {}).setdefault(int(r["question"]), []).append(
            (float(r["quality"]), float(r["individual"]), float(r["self_bleu"])),
        )
    rng = np.random.default_rng(seed)
    rows = []
    for task, label, a, b in pairs:
        if (task, a) not in values or (task, b) not in values:
            continue
        va, vb = values[(task, a)], values[(task, b)]
        shared = sorted(set(va) & set(vb))
        if not shared:
            continue
        xa = np.array([np.mean(va[q], axis=0) for q in shared])
        xb = np.array([np.mean(vb[q], axis=0) for q in shared])
        diff = xa - xb
        boot = diff[rng.integers(0, len(shared), size=(samples, len(shared)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        row: dict[str, Any] = {
            "task": task,
            "comparison": label,
            "a_tag": a[2],
            "b_tag": b[2],
            "questions": len(shared),
        }
        for j, (name, scale) in enumerate((("quality", 100), ("individual", 100), ("self_bleu", 1))):
            row[f"{name}_a"] = scale * xa[:, j].mean()
            row[f"{name}_b"] = scale * xb[:, j].mean()
            row[f"{name}_diff"] = scale * diff[:, j].mean()
            row[f"{name}_lo"] = scale * lo[j]
            row[f"{name}_hi"] = scale * hi[j]
        rows.append(row)
    return rows


def cmd_select(args: argparse.Namespace) -> int:
    all_sources = sources()
    chosen = [all_sources[i] for i in parse_indices(args.index, len(all_sources))] if args.index else all_sources
    features_root = args.features_root or args.out
    jobs = [(source, args.waves_root, features_root, args.gram) for source in chosen]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        source_records = [r for chunk in pool.map(_source_records, jobs) for r in chunk]
        if not source_records:
            print(f"No features under {features_root / 'features'}", file=sys.stderr)
            return 1
        tasks = {r["task"] for r in source_records}
        others = [(source, args.waves_root) for source in counterparts(tasks)]
        other_records = [r for chunk in pool.map(_counterpart_records, others) for r in chunk]
    # Search runs only on the questions the reranked pools cover (smoke runs, partial waves).
    covered = {(r["task"], int(r["question"])) for r in source_records}
    other_records = [r for r in other_records if (r["task"], int(r["question"])) in covered]
    records = source_records + other_records
    fronts.write_csv(args.out / "per_question.csv", records, fronts.PER_QUESTION_FIELDS)

    runs = [s.run for s in chosen] + [s.run for s in counterparts(tasks)]
    runs += [pseudo for s in chosen for pseudo in pseudo_runs(s.run)]
    points, front_rows = fronts.summarize(records, runs)
    fronts.write_csv(args.out / "points.csv", points)
    fronts.write_csv(args.out / "fronts.csv", front_rows)
    paired = paired_rows(records, comparison_pairs(s.run for s in chosen))
    fronts.write_csv(args.out / "paired.csv", paired)
    print(
        f"Wrote {len(records)} per-question rows, {len(points)} points, {len(front_rows)} fronts and "
        f"{len(paired)} paired comparisons to {args.out}",
    )
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------


def parse_indices(spec: str, count: int) -> list[int]:
    """'all', '3', '0-4,7' -> sorted indices within [0, count)."""
    if spec == "all":
        return list(range(count))
    indices: set[int] = set()
    for part in spec.split(","):
        start, _, end = part.partition("-")
        indices.update(range(int(start), int(end or start) + 1))
    assert all(0 <= i < count for i in indices), f"Indices {spec} outside [0, {count - 1}]"
    return sorted(indices)


def cmd_list(_args: argparse.Namespace) -> int:
    for index, source in enumerate(sources()):
        task = source.run.task
        print(f"{index:3d}  {source.name:60s}  pool {task.pool_size:3d} = {task.n_groups} x {task.group_size}")
    return 0


def cmd_count(_args: argparse.Namespace) -> int:
    print(len(sources()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m d5p4.waves.posthoc", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("count")
    default_root = Path(os.environ.get("WAVES_ROOT", "."))
    for name in ("features", "select"):
        p = sub.add_parser(name)
        p.add_argument("--waves-root", type=Path, default=default_root, help="Root holding the source waves.")
        p.add_argument("--out", type=Path, default=default_root / "a1_posthoc", help="A1 output root.")
        p.add_argument(
            "--index",
            default="all" if name == "select" else None,
            required=name == "features",
            help="Source indices (see `list`): 'all', '3', '0-4,7'.",
        )
    p = sub.choices["features"]
    p.add_argument("--config", type=Path, required=True, help="YAML config, e.g. src/d5p4/_jz.yaml.")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="First N questions per source (smoke runs).")
    p.add_argument("--rows-per-forward", type=int, default=ROWS_PER_FORWARD)
    p.add_argument("--force", action="store_true", help="Recompute sources whose features exist.")
    p = sub.choices["select"]
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--features-root", type=Path, default=None, help="Root holding features/ (default: --out).")
    p.add_argument(
        "--gram",
        choices=("gram", "gram_content"),
        default="gram",
        help="D5P4's kernel (default) or the answer-token-only robustness variant; write it to another --out.",
    )
    args = parser.parse_args(argv)
    commands = {"list": cmd_list, "count": cmd_count, "features": cmd_features, "select": cmd_select}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

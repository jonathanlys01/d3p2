"""Paired summaries for the open-ended (MDLM) waves: perplexity, diversity and MAUVE fronts, with tests per prompt.

Every run of a task generates from the same held-out prompts, so methods are compared paired by prompt: a bootstrap
resamples prompts, and each resampled prompt carries all its seeds (seeds nested in prompts), as in
`d5p4.waves.fronts`. Per prompt and run, the k = n_groups texts of a search method form one set; independent sampling
keeps a pool of N = n_groups * group_size and is scored as a uniformly random k-subset of it: perplexity and
self-cosine as exact expectations over k-subsets (pool means), Self-BLEU averaged over a random partition of the pool
into N / k sets, and MAUVE on its first k texts (the same number of texts as every other point).

Per point (one knob value of one method, seeds pooled): GPT-2 Large perplexity (exp of the mean per-text NLL),
self-cosine (mean pairwise Jina cosine within a set), Self-BLEU within a set, unigram token entropy, and MAUVE / MAUVE*
of all its texts against the held-out reference continuations, each with a 95% bootstrap interval and its paired
difference to independent sampling at T = 1.

Per front (a method swept over its knob): hypervolume in the (self-cosine, log perplexity) and (Self-BLEU, log
perplexity) planes, both lower-is-better, as the share of the task's box that the front dominates, and the best MAUVE
along the front (the maximum is recomputed in every bootstrap replicate). `comparisons` then tests D5P4 (additive
kernel) against every other front on these statistics: the paired difference, its 95% interval, a two-sided bootstrap
p-value, and Holm's adjustment over all comparisons of the task. The grid is fixed in the wave definition, so nothing
is selected on the prompts that are reported.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from d5p4.eval_utils import _compute_self_bleu_impl
from d5p4.mauve import compute_mauve
from d5p4.waves.fronts import _interval, _result_file, base_tag, hypervolume, write_csv
from d5p4.waves.open_eval import FEATURES, REFERENCE_DIR, flatten_run
from d5p4.waves.spec import Run


REFERENCE_TAG = "ind_t1"
FOCUS_FRONT = "d5p4"
METRICS = ("nll", "self_cos", "self_bleu", "entropy")
PLANES = {"cos": "self_cos", "bleu": "self_bleu"}
BOOTSTRAP_SAMPLES = 2000
MAUVE_BOOTSTRAP_SAMPLES = 100
PER_PROMPT_FIELDS = ["task", "tag", "front", "run", "prompt", *METRICS]


def front_name(tag: str) -> str:
    """Front of a point: the method prefix of its tag (ind, gbs, dbs, d5p4, d5p4x)."""
    return tag.split("_")[0]


def run_param(run: Run) -> float | None:
    overrides = run.overrides
    return overrides.get("_w_interaction", overrides.get("_diversity_alpha", overrides.get("cat_temperature")))


def self_cosine(embeddings: np.ndarray) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    gram = embeddings @ embeddings.T
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def prompt_sets(
    slots: np.ndarray,
    k: int,
    independent: bool,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Row sets scored for Self-BLEU, and the rows MAUVE uses, among one prompt's rows (indexed by `slots`).

    A search run has exactly k rows: one set, all used by MAUVE. An independent pool of N rows is split at random
    into N // k disjoint sets of k, and MAUVE uses the rows of slots 0..k-1.
    """
    rows = np.argsort(slots, kind="stable")
    if not independent:
        return [rows], rows
    perm = rng.permutation(rows)
    sets = [perm[i : i + k] for i in range(0, len(perm) - len(perm) % k, k)] or [rows]
    return sets, rows[slots[rows] < k]


def run_prompt_records(
    run: Run,
    features: dict[str, np.ndarray],
    texts: list[str],
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Per-prompt metrics of one run, and per prompt the MAUVE feature rows of the texts MAUVE uses."""
    k = run.task.k
    independent = run.method == "baseline"
    records, mauve_rows = [], {}
    for prompt in np.unique(features["prompt"]):
        index = np.flatnonzero(features["prompt"] == prompt)
        rng = np.random.default_rng([seed, int(prompt)])
        sets, mauve_index = prompt_sets(features["slot"][index], k, independent, rng)
        mauve_rows[int(prompt)] = features["mauve"][index[mauve_index]]
        records.append(
            {
                "task": run.task.name,
                "tag": base_tag(run.tag),
                "front": front_name(run.tag),
                "run": run.tag,
                "prompt": int(prompt),
                "nll": float(features["nll"][index].mean()),
                "self_cos": self_cosine(features["jina"][index]),
                "self_bleu": float(np.mean([_compute_self_bleu_impl([texts[index[i]] for i in s]) for s in sets])),
                "entropy": float(features["entropy"][index].mean()),
            },
        )
    return records, mauve_rows


@dataclass
class TaskData:
    task: str
    points: list[str]  # base tags
    fronts: list[str]
    params: list[float | None]
    prompts: list[int]
    metrics: dict[str, np.ndarray]  # metric -> points x prompts (seeds averaged)
    mauve: list[dict[int, np.ndarray]]  # per point, prompt -> stacked feature rows of every seed
    reference: np.ndarray  # reference MAUVE features
    reference_metrics: dict[str, float]


def load_task(runs: list[Run], root: Path) -> tuple[TaskData | None, list[dict[str, Any]]]:
    task = runs[0].task
    eval_dir = root / "eval" / task.name
    reference_path = eval_dir / REFERENCE_DIR / FEATURES
    records: list[dict[str, Any]] = []
    mauve: dict[str, dict[int, list[np.ndarray]]] = {}
    params: dict[str, float | None] = {}
    for run in runs:
        feature_path = eval_dir / run.tag / FEATURES
        result = _result_file(str(run.results_dir(root)))
        if result is None or not feature_path.exists():
            continue
        texts, _, _ = flatten_run(json.loads(Path(result).read_text()))
        features = dict(np.load(feature_path))
        run_records, rows = run_prompt_records(run, features, texts)
        records.extend(run_records)
        params[base_tag(run.tag)] = run_param(run)
        for prompt, feats in rows.items():
            mauve.setdefault(base_tag(run.tag), {}).setdefault(prompt, []).append(feats)
    if not records or not reference_path.exists():
        return None, records
    points = sorted({r["tag"] for r in records})
    by_point: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for r in records:
        by_point.setdefault(r["tag"], {}).setdefault(r["prompt"], []).append(r)
    prompts = sorted(set.intersection(*(set(v) for v in by_point.values())))
    metrics = {
        m: np.array([[np.mean([r[m] for r in by_point[p][q]]) for q in prompts] for p in points]) for m in METRICS
    }
    reference = dict(np.load(reference_path))
    data = TaskData(
        task=task.name,
        points=points,
        fronts=[front_name(p) for p in points],
        params=[params[p] for p in points],
        prompts=prompts,
        metrics=metrics,
        mauve=[{q: np.concatenate(mauve[p][q]) for q in prompts} for p in points],
        reference=reference["mauve"],
        reference_metrics={
            "perplexity": float(np.exp(reference["nll"].mean())),
            "nll": float(reference["nll"].mean()),
            "entropy": float(reference["entropy"].mean()),
        },
    )
    return data, records


# MAUVE ------------------------------------------------------------------------------------------------------------

_MAUVE_STATE: dict[str, Any] = {}


def mauve_score(reference: np.ndarray, generated: np.ndarray, seed: int = 25) -> tuple[float, float]:
    """MAUVE and MAUVE* with the package defaults (k-means buckets: a tenth of the smaller set)."""
    out = compute_mauve(p_features=reference, q_features=generated, seed=seed, verbose=False)
    return float(out.mauve), float(out.mauve_star)


def _point_features(point: dict[int, np.ndarray], prompts: list[int], sample: np.ndarray | None) -> np.ndarray:
    chosen = prompts if sample is None else [prompts[i] for i in sample]
    return np.concatenate([point[q] for q in chosen])


def _mauve_job(job: tuple[int, int]) -> tuple[int, int, float, float]:
    replicate, point = job
    state = _MAUVE_STATE
    sample = None if replicate < 0 else state["samples"][replicate]
    generated = _point_features(state["mauve"][point], state["prompts"], sample)
    return (replicate, point, *mauve_score(state["reference"], generated))


def _init_worker() -> None:
    """One faiss thread per process: the bootstrap parallelizes over processes, and a multithreaded faiss next to
    torch's OpenMP runtime crashes on macOS."""
    faiss.omp_set_num_threads(1)


def mauve_table(data: TaskData, samples: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """MAUVE and MAUVE* as (points, 1 + replicates) arrays: column 0 on every prompt, then one per bootstrap sample."""
    _MAUVE_STATE.update(mauve=data.mauve, prompts=data.prompts, reference=data.reference, samples=samples)
    jobs = [(b, p) for b in range(-1, len(samples)) for p in range(len(data.points))]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(_mauve_job, jobs, chunksize=1)
    else:
        _init_worker()
        results = [_mauve_job(job) for job in jobs]
    mauve = np.zeros((len(data.points), len(samples) + 1))
    star = np.zeros_like(mauve)
    for replicate, point, value, value_star in results:
        mauve[point, replicate + 1] = value
        star[point, replicate + 1] = value_star
    return mauve, star


# Summaries --------------------------------------------------------------------------------------------------------


def bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value of a paired difference being zero, floored at 1 / samples."""
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return float("nan")
    p = 2 * min(np.mean(finite <= 0), np.mean(finite >= 0))
    return float(min(1.0, max(p, 1.0 / len(finite))))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaNs pass through)."""
    finite = [(p, i) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = list(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(finite)):
        running = max(running, min(1.0, (len(finite) - rank) * p))
        adjusted[i] = running
    return adjusted


def _front_hypervolumes(
    data: TaskData,
    x: np.ndarray,
    nll: np.ndarray,
    x_boot: np.ndarray,
    nll_boot: np.ndarray,
) -> dict[str, tuple[float, np.ndarray]]:
    """Normalized hypervolume per front: lower x and lower NLL are better (NLL enters as quality = -NLL)."""
    x_ref, y_ref = float(x.max()), float(-nll.max())
    area = (x_ref - float(x.min())) * (float(-nll.min()) - y_ref)
    result = {}
    for front in dict.fromkeys(data.fronts):
        members = [i for i, f in enumerate(data.fronts) if f == front]
        if area <= 0:
            result[front] = (float("nan"), np.full(x_boot.shape[1], np.nan))
            continue
        value = hypervolume(x[members], -nll[members], x_ref, y_ref) / area
        boot = np.array(
            [
                hypervolume(x_boot[members, b], -nll_boot[members, b], x_ref, y_ref) / area
                for b in range(x_boot.shape[1])
            ],
        )
        result[front] = (value, boot)
    return result


def _point_rows(
    data: TaskData,
    means: dict[str, np.ndarray],
    boots: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """One row per point: every metric with its interval, and its paired difference to independent sampling at T 1."""
    ref = data.points.index(REFERENCE_TAG) if REFERENCE_TAG in data.points else None
    rows = []
    for i, point in enumerate(data.points):
        nll_lo, nll_hi = _interval(boots["nll"][i])
        row: dict[str, Any] = {
            "task": data.task,
            "tag": point,
            "front": data.fronts[i],
            "param_value": data.params[i],
            "prompts": len(data.prompts),
            "perplexity": float(np.exp(means["nll"][i])),
            "perplexity_lo": float(np.exp(nll_lo)),
            "perplexity_hi": float(np.exp(nll_hi)),
        }
        for m in ("self_cos", "self_bleu", "entropy"):
            row[m] = float(means[m][i])
            row[f"{m}_lo"], row[f"{m}_hi"] = _interval(boots[m][i])
        for name, table in tables.items():
            row[name] = float(table[i, 0])
            row[f"{name}_lo"], row[f"{name}_hi"] = _interval(table[i, 1:])
        if ref is not None:
            differences = {m: (means[m][i] - means[m][ref], boots[m][i] - boots[m][ref]) for m in METRICS}
            for name, table in tables.items():
                differences[name] = (table[i, 0] - table[ref, 0], table[i, 1:] - table[ref, 1:])
            for name, (value, values) in differences.items():
                row[f"{name}_minus_ref"] = float(value)
                row[f"{name}_minus_ref_lo"], row[f"{name}_minus_ref_hi"] = _interval(values)
        rows.append(row)
    rows.append(
        {
            "task": data.task,
            "tag": "human_reference",
            "front": "human",
            "perplexity": data.reference_metrics["perplexity"],
            "entropy": data.reference_metrics["entropy"],
        },
    )
    return rows


def _front_statistics(
    data: TaskData,
    means: dict[str, np.ndarray],
    boots: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
) -> dict[str, dict[str, tuple[float, np.ndarray]]]:
    """Per statistic and front: its value on every prompt and its bootstrap values."""
    statistics = {
        f"hv_{plane}": _front_hypervolumes(data, means[metric], means["nll"], boots[metric], boots["nll"])
        for plane, metric in PLANES.items()
    }
    for name, table in tables.items():
        statistics[f"max_{name}"] = {}
        for front in dict.fromkeys(data.fronts):
            members = [i for i, f in enumerate(data.fronts) if f == front]
            statistics[f"max_{name}"][front] = (float(table[members, 0].max()), table[members, 1:].max(axis=0))
    return statistics


def _comparisons(task: str, statistics: dict[str, dict[str, tuple[float, np.ndarray]]]) -> list[dict[str, Any]]:
    """D5P4 minus every other front on every statistic, with paired intervals and Holm-adjusted p-values."""
    rows = []
    for name, per_front in statistics.items():
        if FOCUS_FRONT not in per_front:
            continue
        focus_value, focus_boot = per_front[FOCUS_FRONT]
        for other, (value, values) in per_front.items():
            if other == FOCUS_FRONT:
                continue
            n_boot = min(len(focus_boot), len(values))
            diff = focus_boot[:n_boot] - values[:n_boot]
            lo, hi = _interval(diff)
            rows.append(
                {
                    "task": task,
                    "statistic": name,
                    "front": FOCUS_FRONT,
                    "baseline": other,
                    "difference": focus_value - value,
                    "lo": lo,
                    "hi": hi,
                    "p_bootstrap": bootstrap_p(diff),
                    "bootstrap_samples": n_boot,
                },
            )
    for row, adjusted in zip(rows, holm([r["p_bootstrap"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
    return rows


def summarize_task(
    data: TaskData,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    mauve_samples: int = MAUVE_BOOTSTRAP_SAMPLES,
    workers: int = 1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Point rows, front rows and D5P4-vs-baseline comparison rows of one task."""
    rng = np.random.default_rng(seed)
    n = len(data.prompts)
    boot = rng.integers(0, n, size=(samples, n))
    means = {m: v.mean(axis=1) for m, v in data.metrics.items()}
    boots = {m: v[:, boot].mean(axis=2) for m, v in data.metrics.items()}  # points x samples
    mauve, mauve_star = mauve_table(data, boot[:mauve_samples], workers)
    tables = {"mauve": mauve, "mauve_star": mauve_star}

    statistics = _front_statistics(data, means, boots, tables)
    front_rows = []
    for front in dict.fromkeys(data.fronts):
        row: dict[str, Any] = {"task": data.task, "front": front, "points": data.fronts.count(front)}
        for name, per_front in statistics.items():
            value, values = per_front[front]
            row[name] = value
            row[f"{name}_lo"], row[f"{name}_hi"] = _interval(values)
        front_rows.append(row)
    return _point_rows(data, means, boots, tables), front_rows, _comparisons(data.task, statistics)


def default_workers() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def write_outputs(
    runs: list[Run],
    root: Path,
    *,
    mauve_samples: int = MAUVE_BOOTSTRAP_SAMPLES,
    workers: int | None = None,
) -> int:
    """Per-prompt, point, front and comparison CSVs of every open task, in ROOT/fronts/open_*.csv."""
    workers = workers or default_workers()
    all_records, all_points, all_fronts, all_comparisons = [], [], [], []
    for name in dict.fromkeys(run.task.name for run in runs if run.task.family == "open"):
        data, records = load_task([run for run in runs if run.task.name == name], root)
        all_records.extend(records)
        if data is None:
            print(f"Skipping {name}: no featurized runs or no reference features")
            continue
        points, fronts_rows, comparisons = summarize_task(data, mauve_samples=mauve_samples, workers=workers)
        all_points.extend(points)
        all_fronts.extend(fronts_rows)
        all_comparisons.extend(comparisons)
    out = root / "fronts"
    if all_records:
        write_csv(out / "open_per_prompt.csv", all_records, PER_PROMPT_FIELDS)
    for filename, rows in (
        ("open_points.csv", all_points),
        ("open_fronts.csv", all_fronts),
        ("open_comparisons.csv", all_comparisons),
    ):
        if rows:
            write_csv(out / filename, rows)
    print(f"Wrote {len(all_records)} per-prompt rows, {len(all_points)} points, {len(all_fronts)} fronts to {out}")
    return len(all_records)

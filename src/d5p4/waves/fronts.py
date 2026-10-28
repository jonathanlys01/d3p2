"""Per-question metrics and paired summaries for quality-diversity fronts.

Every method of a task runs on the same questions, so comparisons are paired by question: resampling questions in a
bootstrap keeps question difficulty shared between methods and gives intervals on differences that are far tighter
than the per-method standard errors.

Per question and selection rule, a run contributes:

- quality: correct-at-k (math, code) or max F1 over the k answers (QA), in [0, 1];
- individual: mean correctness or F1 of the k answers;
- self_bleu: Self-BLEU between the k answers, as in the evaluator.

Search methods return one answer per group, picked by the internal score. The independent pool is scored two ways:
"random k of N" as the exact expectation over all k-subsets of the pool (Self-BLEU as a Monte Carlo average), which
removes the noise of drawing one subset, and "internal score" as the top k by internal score (internal-score
best-of-N).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections.abc import Iterable
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from d5p4.eval_utils import _compute_f1_score, _compute_self_bleu_impl
from d5p4.waves.spec import Run


REFERENCE_TAG = "ind_t1"
REFERENCE_SELECTION = "random k of N"
SELF_BLEU_SUBSETS = 24
BOOTSTRAP_SAMPLES = 2000
SHARD_SUFFIX = re.compile(r"_s\d+of\d+$")

PER_QUESTION_FIELDS = ["task", "tag", "method", "selection", "question", "quality", "individual", "self_bleu"]


def base_tag(tag: str) -> str:
    """Tag shared by every question shard of one run."""
    return SHARD_SUFFIX.sub("", tag)


def expected_max_of_subset(values: Iterable[float], k: int) -> float:
    """Exact expectation of max(values over a uniformly random k-subset); with 0/1 values it is the unbiased pass@k."""
    v = np.sort(np.asarray(list(values), dtype=float))
    n = len(v)
    k = min(k, n)
    weights = np.array([comb(i, k - 1) for i in range(n)], dtype=float) / comb(n, k)
    return float((v * weights).sum())


def _result_file(run_dir: str) -> str | None:
    files = [
        path
        for path in glob.glob(os.path.join(run_dir, "*.json"))
        if not os.path.basename(path).startswith("temp") and "-bon-" not in os.path.basename(path)
    ]
    return max(files, key=os.path.getmtime) if files else None


def _question_rows(run: Run, data: dict[str, Any]) -> Iterable[tuple[int, list[str], list[float], list[float]]]:
    """Yield (question index, candidate texts, candidate quality in [0, 1], internal scores) for each question."""
    texts = data["text_samples"]
    scores = data.get("internal_scores") or [[0.0] * len(group) for group in texts]
    if (data.get("internal_score_metadata") or {}).get("higher_is_better", True) is False:
        scores = [[-s for s in group] for group in scores]
    indices = data.get("dataset_indices") or list(range(len(texts)))
    family = run.task.family
    if family == "qa":
        references = data["references"]
        for index, cands, refs, internal in zip(indices, texts, references, scores, strict=True):
            f1 = [max((_compute_f1_score(c, r) for r in refs), default=0.0) for c in cands]
            yield index, cands, f1, internal
    else:
        rows = data["results"]["results"] if isinstance(data["results"], dict) else data["results"]
        for index, cands, row, internal in zip(indices, texts, rows, scores, strict=True):
            yield index, cands, [float(x) for x in row["scores"]], internal


def run_question_metrics(run: Run, data: dict[str, Any], seed: int = 0) -> list[dict[str, Any]]:
    """Per-question metrics of one run under the selection rules that the paired summaries use."""
    k = run.task.k
    group_size = run.all_overrides()["group_size"]
    rng = np.random.default_rng(seed)
    records = []
    for index, cands, quality, internal in _question_rows(run, data):
        q = np.asarray(quality, dtype=float)
        s = np.asarray(internal, dtype=float)
        if run.method == "baseline":
            top = list(np.argsort(-s, kind="stable")[:k])
            subsets = [rng.choice(len(cands), k, replace=False) for _ in range(SELF_BLEU_SUBSETS)]
            selections = {
                "random k of N": (
                    expected_max_of_subset(q, k),
                    float(q.mean()),
                    float(np.mean([_compute_self_bleu_impl([cands[i] for i in sub]) for sub in subsets])),
                ),
                "internal score": (
                    float(q[top].max()),
                    float(q[top].mean()),
                    _compute_self_bleu_impl([cands[i] for i in top]),
                ),
            }
        else:
            picks = [
                start + int(np.argmax(s[start : start + group_size])) for start in range(0, len(cands), group_size)
            ]
            selections = {
                "internal score": (
                    float(q[picks].max()),
                    float(q[picks].mean()),
                    _compute_self_bleu_impl([cands[i] for i in picks]),
                ),
            }
        for selection, (quality_k, individual, self_bleu) in selections.items():
            records.append(
                {
                    "task": run.task.name,
                    "tag": base_tag(run.tag),
                    "method": run.method,
                    "selection": selection,
                    "question": index,
                    "quality": quality_k,
                    "individual": individual,
                    "self_bleu": self_bleu,
                },
            )
    return records


def per_question(runs: list[Run], root: Path) -> list[dict[str, Any]]:
    records = []
    for run in runs:
        path = _result_file(str(run.results_dir(root)))
        if path is None:
            continue
        with open(path) as f:
            records.extend(run_question_metrics(run, json.load(f)))
    return records


def _point_matrix(records: list[dict[str, Any]]) -> tuple[list[tuple[str, str, str]], np.ndarray, np.ndarray]:
    """Points (method, selection, tag) as rows and the questions every point answered as columns."""
    by_point: dict[tuple[str, str, str], dict[int, tuple[float, float]]] = {}
    for r in records:
        by_point.setdefault((r["method"], r["selection"], r["tag"]), {})[int(r["question"])] = (
            float(r["quality"]),
            float(r["self_bleu"]),
        )
    points = sorted(by_point)
    shared = sorted(set.intersection(*(set(v) for v in by_point.values())))
    quality = np.array([[by_point[p][q][0] for q in shared] for p in points])
    self_bleu = np.array([[by_point[p][q][1] for q in shared] for p in points])
    return points, quality, self_bleu


def hypervolume(self_bleu: np.ndarray, quality: np.ndarray, x_ref: float, y_ref: float) -> float:
    """Area of the (Self-BLEU, quality) region the points dominate, bounded by the reference corner.

    Lower Self-BLEU and higher quality are better, so a point (x, y) dominates [x, x_ref] x [y_ref, y]. Points beyond
    the corner add nothing.
    """
    keep = (self_bleu < x_ref) & (quality > y_ref)
    xs, ys = self_bleu[keep], quality[keep]
    if len(xs) == 0:
        return 0.0
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])
    widths = np.diff(np.append(xs, x_ref))
    return float((widths * (ys - y_ref)).sum())


def _interval(values: np.ndarray) -> tuple[float, float]:
    """95% percentile interval, or NaNs when most bootstrap samples are undefined."""
    finite = values[np.isfinite(values)]
    if len(finite) < 0.5 * len(values):
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _front_rows(
    task: str,
    point_fronts: list[tuple[str, str, str]],
    means: tuple[np.ndarray, np.ndarray],
    boots: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    """Hypervolume of each front of one task, and its paired differences to the reference fronts."""
    (s_mean, q_mean), (s_boot, q_boot) = means, boots
    x_ref, y_ref = float(s_mean.max()), float(q_mean.min())
    area = (x_ref - float(s_mean.min())) * (float(q_mean.max()) - y_ref)
    fronts_of_task = sorted(set(point_fronts))
    hv = {}
    for key in fronts_of_task:
        members = [i for i, front in enumerate(point_fronts) if front == key]
        hv[key] = (
            hypervolume(s_mean[members], q_mean[members], x_ref, y_ref) / area,
            np.array(
                [
                    hypervolume(s_boot[members, b], q_boot[members, b], x_ref, y_ref) / area
                    for b in range(s_boot.shape[1])
                ],
            ),
        )
    references = {
        "independent": ("baseline", REFERENCE_SELECTION, ""),
        "d5p4": ("greedy_map", "internal score", "1"),
    }
    rows = []
    for key in fronts_of_task:
        value, boot_values = hv[key]
        row = {
            "task": task,
            "method": key[0],
            "selection": key[1],
            "temperature": key[2],
            "points": point_fronts.count(key),
            "hypervolume": value,
            "hv_lo": _interval(boot_values)[0],
            "hv_hi": _interval(boot_values)[1],
        }
        for name, other in references.items():
            if other in hv:
                other_value, other_boot = hv[other]
                lo, hi = _interval(boot_values - other_boot)
                row.update(
                    {f"hv_minus_{name}": value - other_value, f"hv_minus_{name}_lo": lo, f"hv_minus_{name}_hi": hi},
                )
        rows.append(row)
    return rows


def summarize(
    records: list[dict[str, Any]],
    runs: list[Run],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Paired bootstrap summaries per task: one row per point and one row per front.

    Points: mean quality and Self-BLEU, and quality minus the reference point (independent sampling at T = 1.0,
    random k of N) on the same questions, with 95% intervals.

    Fronts: each front's hypervolume, the share of the task's (Self-BLEU, quality) box that it dominates, and its
    difference to the independent-sampling front and to the D5P4 front at T = 1.0, with paired intervals. A front is
    a (method, selection) pair; search methods are further split by sampling temperature, since their fronts sweep
    alpha or beta at one temperature while independent sampling's front sweeps the temperature itself. The box spans
    every point of the task, from the most to the least diverse in Self-BLEU and from the lowest quality to the
    highest. Hypervolume needs no shared Self-BLEU range, unlike quality at matched diversity: the search fronts sit
    entirely at lower Self-BLEU than independent sampling on GSM8K and MBPP.
    """
    params, temperatures = {}, {}
    for run in runs:
        overrides = run.overrides
        value = overrides.get("_w_interaction", overrides.get("_diversity_alpha", overrides.get("cat_temperature")))
        params[(run.task.name, base_tag(run.tag))] = value
        temperatures[(run.task.name, base_tag(run.tag))] = f"{run.all_overrides()['cat_temperature']:g}"
    point_rows, front_rows = [], []
    rng = np.random.default_rng(seed)
    for task in sorted({r["task"] for r in records}):
        points, quality, self_bleu = _point_matrix([r for r in records if r["task"] == task])
        reference = points.index(("baseline", REFERENCE_SELECTION, REFERENCE_TAG))
        n = quality.shape[1]
        boot = rng.integers(0, n, size=(samples, n))
        q_boot = quality[:, boot].mean(axis=2)  # points x samples
        s_boot = self_bleu[:, boot].mean(axis=2)
        q_mean, s_mean = quality.mean(axis=1), self_bleu.mean(axis=1)
        diff_boot = q_boot - q_boot[reference]
        for i, (method, selection, tag) in enumerate(points):
            lo, hi = _interval(diff_boot[i])
            point_rows.append(
                {
                    "task": task,
                    "method": method,
                    "selection": selection,
                    "tag": tag,
                    "param_value": params.get((task, tag)),
                    "temperature": temperatures.get((task, tag), ""),
                    "questions": n,
                    "quality": 100 * q_mean[i],
                    "self_bleu": s_mean[i],
                    "self_bleu_lo": _interval(s_boot[i])[0],
                    "self_bleu_hi": _interval(s_boot[i])[1],
                    "diff_vs_ref": 100 * (q_mean[i] - q_mean[reference]),
                    "diff_lo": 100 * lo,
                    "diff_hi": 100 * hi,
                },
            )
        point_fronts = [
            (method, selection, "" if method == "baseline" else temperatures.get((task, tag), ""))
            for method, selection, tag in points
        ]
        front_rows.extend(_front_rows(task, point_fronts, (s_mean, q_mean), (s_boot, q_boot)))
    return point_rows, front_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

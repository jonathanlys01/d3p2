"""Offline analyses of selection dumps (revision item A3; see `d5p4.selection_dump` for the format).

python -m d5p4.selection_dump_analysis solver  DUMP_DIR [--out OUT]
python -m d5p4.selection_dump_analysis proxy   DUMP_DIR --results RESULT_JSON [--out OUT]
python -m d5p4.selection_dump_analysis theory  [--instances N] [--out OUT]

`solver`: at every dumped step, the exact transversal MAP by enumeration of all w^k sets (27 / 256 / 65,536 at
3 x 3, 4 x 4, 8 x 4), against single-start greedy (Chen et al. 2018 under the partition constraint), multi-start greedy
(one trajectory per start item, the best log det kept, as in production) and the set production actually returned.

Gap metric. The objective f(S) = log det(L_S) can be negative and is not normalized (f of the empty set is 0, but L_S
has eigenvalues below 1), so a ratio f(G) / f(OPT) is meaningless. We report:

- delta = f(OPT) - f(G) >= 0 in nats, equivalently the determinant ratio det(L_G) / det(L_OPT) = exp(-delta), which
  is invariant to rescaling L;
- normalized gap = delta / (f(OPT) - mean_T f(T)): the share of the gain of the best set over an average transversal
  that greedy leaves on the table;
- optimal: the greedy set is the argmax, or ties it within 1e-6 nats;
- rank: the share of the w^k transversals strictly better than the greedy set.

Steps where every transversal has the same value (step 0, where all candidates are the same fully masked state) or
no transversal has a positive determinant carry no information and are counted separately.

`proxy`: Spearman correlation of the step-t quality score with the final quality of the lineage it starts (GSM8K
correctness, QA max-reference F1), binned by the masked fraction of the generation. Lineage = one final evaluated
answer (the group's internal-score pick, as in `d5p4.waves.fronts`) traced back through the selected rows.

`theory`: synthetic CPU checks of the appendix H claims (Neumann / Mercator expansion of det(I + C) at high cosine,
the shared-component variant, and exact vs greedy MAP on structured kernels).
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from d5p4.selection_dump import load_dump


TIE_NATS = 1e-6
MASK_BINS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)


# ── Exact and greedy transversal MAP ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def transversal_sets(n_groups: int, group_size: int) -> np.ndarray:
    """All w^k transversals [w^k, k] for contiguous groups: group g holds rows g * w ... g * w + w - 1."""
    grids = np.meshgrid(*[np.arange(group_size)] * n_groups, indexing="ij")
    local = np.stack([g.reshape(-1) for g in grids], axis=1)
    return local + np.arange(n_groups) * group_size


def logdets(L: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """log det(L_S) for each row of `sets`; -inf where the submatrix is not positive definite."""
    sub = L[sets[:, :, None], sets[:, None, :]].astype(np.float64)
    sign, value = np.linalg.slogdet(sub)
    return np.where(sign > 0, value, -np.inf)


def greedy_transversal(L: np.ndarray, groups: np.ndarray, starts: np.ndarray | None = None) -> np.ndarray:
    """Partition-constrained greedy MAP (Chen et al. 2018), one trajectory per start item.

    `starts=None` runs the plain greedy, whose first pick is the largest diagonal entry. Each later pick maximizes the
    Schur complement d_i^2 (the marginal gain of log det) over items whose group is not yet used. Returns the
    selections [n_starts, k] in pick order.
    """
    L = np.asarray(L, dtype=np.float64)
    n_items = L.shape[0]
    k = int(groups.max()) + 1
    diag = np.diagonal(L).copy()
    if starts is None:
        starts = np.array([int(np.argmax(diag))])
    starts = np.asarray(starts)
    n_traj = len(starts)
    rows = np.arange(n_traj)
    d2 = np.tile(diag, (n_traj, 1))
    basis = np.zeros((n_traj, max(k - 1, 1), n_items))
    used = np.zeros((n_traj, n_items), dtype=bool)
    selected = np.empty((n_traj, k), dtype=np.int64)
    current = starts.copy()
    for t in range(k):
        selected[:, t] = current
        used |= groups[None, :] == groups[current][:, None]
        if t == k - 1:
            break
        dj = np.maximum(d2[rows, current], 1e-300)
        coeffs = basis[rows, :t, current] if t else np.zeros((n_traj, 0))
        e = (L[current] - np.einsum("tj,tjn->tn", coeffs, basis[:, :t])) / np.sqrt(dj)[:, None]
        basis[:, t] = e
        d2 = d2 - e**2
        current = np.argmax(np.where(used, -np.inf, d2), axis=1)
    return selected


def best_of(L: np.ndarray, selections: np.ndarray) -> tuple[np.ndarray, float]:
    values = logdets(L, np.sort(selections, axis=1))
    best = int(np.argmax(values))
    return np.sort(selections[best]), float(values[best])


def solve_step(L: np.ndarray, n_groups: int, group_size: int) -> dict[str, Any]:
    """Exact, single-start and multi-start values of one kernel; the sets are sorted row indices."""
    sets = transversal_sets(n_groups, group_size)
    values = logdets(L, sets)
    finite = np.isfinite(values)
    best = int(np.argmax(values))
    groups = np.repeat(np.arange(n_groups), group_size)
    single_set, single = best_of(L, greedy_transversal(L, groups))
    multi_set, multi = best_of(L, greedy_transversal(L, groups, np.arange(L.shape[0])))
    return {
        "values": values,
        "opt_set": sets[best],
        "opt": float(values[best]),
        "mean": float(values[finite].mean()) if finite.any() else -np.inf,
        "min": float(values[finite].min()) if finite.any() else -np.inf,
        "single_set": single_set,
        "single": single,
        "multi_set": multi_set,
        "multi": multi,
    }


def gap_row(result: dict[str, Any], value: float, chosen: np.ndarray, prefix: str) -> dict[str, Any]:
    opt, mean = result["opt"], result["mean"]
    delta = opt - value
    spread = opt - mean
    return {
        f"{prefix}_logdet": value,
        f"{prefix}_delta": delta,
        f"{prefix}_det_ratio": float(np.exp(-delta)) if np.isfinite(delta) else 0.0,
        f"{prefix}_norm_gap": delta / spread if spread > TIE_NATS else 0.0,
        f"{prefix}_optimal": int(delta <= TIE_NATS or np.array_equal(chosen, result["opt_set"])),
        f"{prefix}_rank": float((result["values"] > value + TIE_NATS).mean()),
    }


def fnw_bound(L: np.ndarray, opt: float, k: int) -> float:
    """Lower bound on log det(L_G) for plain greedy from Fisher, Nemhauser & Wolsey (1978).

    With lambda = lambda_min(L) > 0, g(S) = log det(L_S / lambda) is normalized, monotone and submodular, and greedy
    picks the same items on L / lambda as on L, so g(G) >= g(OPT) / 2 over the bases of the partition matroid:
    log det(L_G) >= log det(L_OPT) / 2 + (k / 2) log lambda. Vacuous (-inf) when L is singular.
    """
    lam = float(np.linalg.eigvalsh(np.asarray(L, dtype=np.float64))[0])
    return 0.5 * opt + 0.5 * k * np.log(lam) if lam > 0 else -np.inf


def expansion_norms(S_sub: np.ndarray) -> tuple[float, float, float]:
    """||C||_2 for C = S_sub - I (appendix H.1), and min over rho of ||H(rho)||_2 (H.2), with the minimizing rho."""
    k = S_sub.shape[0]
    C = S_sub - np.eye(k)
    c_norm = float(np.linalg.norm(C, 2))
    best_h, best_rho = np.inf, 0.0
    for rho in np.linspace(0.0, 0.995, 200):
        # B_rho^{-1/2}: eigenvalues 1 + (k - 1) rho on the ones direction, 1 - rho on its complement.
        ones = np.ones((k, 1)) / np.sqrt(k)
        proj = ones @ ones.T
        b_inv_half = proj / np.sqrt(1 + (k - 1) * rho) + (np.eye(k) - proj) / np.sqrt(1 - rho)
        H = b_inv_half @ S_sub @ b_inv_half - np.eye(k)
        h = float(np.linalg.norm(H, 2))
        if h < best_h:
            best_h, best_rho = h, float(rho)
    return c_norm, best_h, best_rho


# ── Dumps ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def _dumps(dump_dir: Path) -> list[Path]:
    paths = sorted(p for p in dump_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not paths:
        raise SystemExit(f"No dumps under {dump_dir}")
    return paths


def solver_rows(dump: dict[str, Any], question: str) -> list[dict[str, Any]]:
    meta = dump["meta"]
    k, w = meta["n_groups"], meta["group_size"]
    rows = []
    for L32, S, step in zip(dump["L"], dump["S"], dump["kernel_steps"], strict=True):
        L = L32.astype(np.float64)
        result = solve_step(L, k, w)
        row: dict[str, Any] = {
            "question": question,
            "step": int(step),
            "mask_frac": float(dump["mask_frac"][step]),
            "opt": result["opt"],
            "mean": result["mean"],
            "min": result["min"],
            "degenerate": int(not np.isfinite(result["opt"]) or result["opt"] - result["min"] <= TIE_NATS),
            # Siblings often unmask the same token, so duplicate candidates (and tied sets) are common: compare values.
            "distinct_candidates": len(np.unique(L, axis=0)),
            "single_eq_multi": int(abs(result["single"] - result["multi"]) <= TIE_NATS),
            "single_same_set_multi": int(np.array_equal(result["single_set"], result["multi_set"])),
        }
        row.update(gap_row(result, result["single"], result["single_set"], "single"))
        row.update(gap_row(result, result["multi"], result["multi_set"], "multi"))
        production = np.sort(dump["sel"][step].astype(np.int64))
        row.update(gap_row(result, float(logdets(L, production[None])[0]), production, "prod"))
        row["prod_eq_multi"] = int(abs(row["prod_logdet"] - result["multi"]) <= TIE_NATS)
        bound = fnw_bound(L, result["opt"], k) if np.isfinite(result["opt"]) else -np.inf
        row["fnw_bound"] = bound
        row["fnw_holds"] = int(result["single"] >= bound - TIE_NATS)
        row["fnw_slack"] = result["single"] - bound
        c_norm, h_norm, rho = expansion_norms(S.astype(np.float64)[np.ix_(production, production)])
        row["opt_mean_cos"] = float(
            np.mean([S[i, j] for i, j in combinations(result["opt_set"], 2)]) if k > 1 else 1.0,
        )
        row["prod_C_norm"], row["prod_H_norm"], row["prod_H_rho"] = c_norm, h_norm, rho
        rows.append(row)
    return rows


def summarize_solver(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live = [r for r in rows if not r["degenerate"]]
    out: dict[str, Any] = {
        "steps": len(rows),
        "degenerate_steps": len(rows) - len(live),
        "informative_steps": len(live),
    }
    if not live:
        return out
    for prefix in ("single", "multi", "prod"):
        delta = np.array([r[f"{prefix}_delta"] for r in live])
        ratio = np.array([r[f"{prefix}_det_ratio"] for r in live])
        norm = np.array([r[f"{prefix}_norm_gap"] for r in live])
        out[prefix] = {
            "optimal_frac": float(np.mean([r[f"{prefix}_optimal"] for r in live])),
            "delta_mean": float(delta.mean()),
            "delta_p50": float(np.percentile(delta, 50)),
            "delta_p99": float(np.percentile(delta, 99)),
            "delta_max": float(delta.max()),
            "det_ratio_min": float(ratio.min()),
            "det_ratio_p1": float(np.percentile(ratio, 1)),
            "norm_gap_mean": float(norm.mean()),
            "rank_mean": float(np.mean([r[f"{prefix}_rank"] for r in live])),
        }
    out["single_eq_multi_frac"] = float(np.mean([r["single_eq_multi"] for r in live]))
    out["single_same_set_multi_frac"] = float(np.mean([r["single_same_set_multi"] for r in live]))
    out["prod_eq_multi_frac"] = float(np.mean([r["prod_eq_multi"] for r in live]))
    out["distinct_candidates_mean"] = float(np.mean([r["distinct_candidates"] for r in live]))
    out["fnw_holds_frac"] = float(np.mean([r["fnw_holds"] for r in live]))
    out["fnw_slack_p50"] = float(np.median([r["fnw_slack"] for r in live]))
    out["prod_C_norm_p50"] = float(np.median([r["prod_C_norm"] for r in live]))
    out["prod_C_norm_lt1_frac"] = float(np.mean([r["prod_C_norm"] < 1 for r in live]))
    out["prod_H_norm_lt1_frac"] = float(np.mean([r["prod_H_norm"] < 1 for r in live]))
    out["opt_mean_cos_p50"] = float(np.median([r["opt_mean_cos"] for r in live]))
    by_bin = {}
    for hi, lo in zip(MASK_BINS[:-1], MASK_BINS[1:], strict=True):
        sel = [r for r in live if lo < r["mask_frac"] <= hi]
        if sel:
            by_bin[f"{lo:.1f}-{hi:.1f}"] = {
                "steps": len(sel),
                "single_optimal_frac": float(np.mean([r["single_optimal"] for r in sel])),
                "multi_optimal_frac": float(np.mean([r["multi_optimal"] for r in sel])),
                "single_delta_mean": float(np.mean([r["single_delta"] for r in sel])),
            }
    out["by_mask_frac"] = by_bin
    return out


def cmd_solver(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in _dumps(args.dump_dir):
        dump = load_dump(path)
        if len(dump["kernel_steps"]) == 0:
            continue
        rows.extend(solver_rows(dump, path.stem))
    if not rows:
        raise SystemExit("No kernels in these dumps (independent sampling dumps carry scores only).")
    out = args.out or args.dump_dir.parent / "a3_solver"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "steps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_solver(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


# ── Quality proxy on partially masked states ──────────────────────────────────────────────────────────────────────


def ancestors(sel: np.ndarray, selected: np.ndarray, group_size: int, final_rows: np.ndarray) -> np.ndarray:
    """Row index at every step [T, len(final_rows)] of the candidates that the final rows descend from."""
    n_steps = len(sel)
    out = np.empty((n_steps, len(final_rows)), dtype=np.int64)
    current = np.asarray(final_rows, dtype=np.int64)
    for t in range(n_steps - 1, -1, -1):
        if selected[t]:
            current = sel[t][current // group_size].astype(np.int64)
        out[t] = current
    return out


def _question_records(results_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """(internal scores, final quality in [0, 1]) per question of one generation JSON, rows in result order."""
    data = json.loads(results_path.read_text())
    texts = data["text_samples"]
    scores = data["internal_scores"]
    records = []
    results = data.get("results")
    if results is not None:
        rows = results["results"] if isinstance(results, dict) else results
        for internal, row in zip(scores, rows, strict=True):
            records.append((np.asarray(internal, float), np.asarray(row["scores"], float)))
        return records
    from d5p4.eval_utils import _compute_f1_score  # noqa: PLC0415  (pulls in transformers; QA only)

    for internal, cands, refs in zip(scores, texts, data["references"], strict=True):
        f1 = [max((_compute_f1_score(c, r) for r in refs), default=0.0) for c in cands]
        records.append((np.asarray(internal, float), np.asarray(f1, float)))
    return records


def _match(final_scores: np.ndarray, records: list[tuple[np.ndarray, np.ndarray]]) -> int | None:
    for index, (internal, _quality) in enumerate(records):
        if internal.shape == final_scores.shape and np.allclose(internal, final_scores, rtol=0, atol=1e-9):
            return index
    return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _concordance(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Sum over lineage pairs with different final quality of 1 if the score orders them right, 0.5 on score ties."""
    total, pairs = 0.0, 0
    for i, j in combinations(range(len(x)), 2):
        if y[i] == y[j]:
            continue
        pairs += 1
        dx, dy = x[i] - x[j], y[i] - y[j]
        total += 0.5 if dx == 0 else float(dx * dy > 0)
    return total, pairs


def proxy_table(lineages: list[dict[str, Any]], score_key: str, rng_seed: int = 0) -> list[dict[str, Any]]:
    """Per masked-fraction bin: pooled Spearman, mean within-question Spearman and pairwise concordance."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    bins = [*zip(MASK_BINS[:-1], MASK_BINS[1:], strict=True), ("final", "final")]
    for hi, lo in bins:
        per_q = []
        pooled_x, pooled_y = [], []
        for item in lineages:
            mask = item["mask_frac"]
            steps = [len(mask) - 1] if hi == "final" else np.nonzero((mask <= hi) & (mask > lo))[0]
            if len(steps) == 0:
                continue
            x = item[score_key][steps].mean(axis=0)
            y = item["quality"]
            pooled_x.extend(x)
            pooled_y.extend(y)
            conc, pairs = _concordance(x, y)
            per_q.append((_spearman(x, y), conc, pairs))
        if not per_q:
            continue
        rho = np.array([r for r, _, _ in per_q])
        conc = np.array([c for _, c, _ in per_q])
        pairs = np.array([p for _, _, p in per_q])
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(per_q), len(per_q))
            if pairs[idx].sum():
                boots.append(conc[idx].sum() / pairs[idx].sum())
        label = "final" if hi == "final" else f"{lo:.1f}-{hi:.1f}"
        rows.append(
            {
                "mask_frac": label,
                "questions": len(per_q),
                "pooled_spearman": _spearman(np.asarray(pooled_x), np.asarray(pooled_y)),
                "within_q_spearman_mean": float(np.nanmean(rho)) if np.isfinite(rho).any() else float("nan"),
                "within_q_spearman_n": int(np.isfinite(rho).sum()),
                "concordance": float(conc.sum() / pairs.sum()) if pairs.sum() else float("nan"),
                "concordance_lo": float(np.percentile(boots, 2.5)) if boots else float("nan"),
                "concordance_hi": float(np.percentile(boots, 97.5)) if boots else float("nan"),
                "pairs": int(pairs.sum()),
            },
        )
    return rows


def cmd_proxy(args: argparse.Namespace) -> int:
    records = _question_records(args.results)
    lineages = []
    unmatched = 0
    for path in _dumps(args.dump_dir):
        dump = load_dump(path)
        index = _match(dump["final_internal_scores"], records)
        if index is None:
            unmatched += 1
            continue
        internal, quality = records[index]
        w = dump["meta"]["group_size"]
        # One evaluated answer per group: the group's internal-score pick, as in the fronts.
        picks = np.array([g + int(np.argmax(internal[g : g + w])) for g in range(0, len(internal), w)])
        anc = ancestors(dump["sel"], dump["selected"], w, picks)
        steps = np.arange(len(anc))
        lineages.append(
            {
                "mask_frac": dump["mask_frac"],
                "quality": quality[picks],
                "neg_entropy": dump["score_neg_entropy"][steps[:, None], anc],
                "confidence": dump["score_confidence"][steps[:, None], anc],
            },
        )
    print(f"Joined {len(lineages)} dumps to questions of {args.results.name}; {unmatched} unmatched.")
    out = args.out or args.dump_dir.parent / "a3_proxy"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key in ("neg_entropy", "confidence"):
        table = proxy_table(lineages, key)
        summary[key] = table
        with (out / f"{key}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"\n{key}")
        for row in table:
            print(
                f"  {row['mask_frac']:>8s}  pooled rho {row['pooled_spearman']:+.3f}  "
                f"within-q rho {row['within_q_spearman_mean']:+.3f} (n={row['within_q_spearman_n']})  "
                f"concordance {row['concordance']:.3f} [{row['concordance_lo']:.3f}, {row['concordance_hi']:.3f}] "
                f"({row['pairs']} pairs)",
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


# ── Synthetic checks of appendix H ────────────────────────────────────────────────────────────────────────────────


def equicorrelated_rows(cosines: tuple[float, ...], ks: tuple[int, ...]) -> list[dict[str, Any]]:
    """H.1 at C = c (11^T - I): exact det(I + C) against the second- and third-order expansions."""
    rows = []
    for k in ks:
        m = k * (k - 1) // 2
        for c in cosines:
            C = c * (np.ones((k, k)) - np.eye(k))
            det = float(np.linalg.det(np.eye(k) + C))
            second = 1 - m * c**2
            third = second + np.trace(C @ C @ C) / 3
            rows.append(
                {
                    "k": k,
                    "cos": c,
                    "C_norm": float(np.linalg.norm(C, 2)),
                    "det_exact": det,
                    "det_order2": second,
                    "det_order3": float(third),
                    "sum_c2": m * c**2,
                    "one_minus_det": 1 - det,
                    "mean_cos": c,
                    "cs_bound": float(np.sqrt(max(1 - det, 0) / m)),
                    "fischer_bound": float(np.sqrt(max(1 - det, 0))),
                },
            )
    return rows


def unit_vectors_with_mean_cosine(k: int, mean_cos: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    """k unit vectors: shared direction with weight sqrt(rho) plus independent Gaussian residuals (cos ~ rho)."""
    shared = rng.standard_normal(dim)
    shared /= np.linalg.norm(shared)
    z = rng.standard_normal((k, dim))
    z -= np.outer(z @ shared, shared)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    x = np.sqrt(mean_cos) * shared + np.sqrt(1 - mean_cos) * z
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def structured_kernel(  # noqa: PLR0913
    n_groups: int,
    group_size: int,
    rng: np.random.Generator,
    lineage_cos: float = 0.9,
    sibling_cos: float | None = 0.99,
    beta: float = 20.0,
    dim: int = 512,
) -> np.ndarray:
    """L = diag(Q) + beta * S with D5P4's structure: siblings (one group, one parent) far closer than lineages.

    Lineage centres share a component with weight `lineage_cos` (pairwise cosine about that value); siblings add a
    small private perturbation so their cosine to each other is about `sibling_cos`. Q is uniform, min-max normalized
    to [0, 1] as in production. `sibling_cos=None` drops the group structure (i.i.d. candidates).
    """
    if sibling_cos is None:
        # Unstructured: every candidate is its own lineage, groups are an arbitrary partition.
        x = unit_vectors_with_mean_cosine(n_groups * group_size, lineage_cos, dim, rng)
        q = rng.random(n_groups * group_size)
        return beta * (x @ x.T) + np.diag((q - q.min()) / (q.max() - q.min()))
    parents = unit_vectors_with_mean_cosine(n_groups, lineage_cos, dim, rng)
    noise = rng.standard_normal((n_groups * group_size, dim))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    eps = np.sqrt(1 - sibling_cos)
    x = np.repeat(parents, group_size, axis=0) * np.sqrt(sibling_cos) + eps * noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    S = x @ x.T
    q = rng.random(n_groups * group_size)
    q = (q - q.min()) / (q.max() - q.min())
    return beta * S + np.diag(q)


def synthetic_solver(  # noqa: PLR0913
    n_groups: int,
    group_size: int,
    instances: int,
    rng: np.random.Generator,
    **kernel_kwargs: Any,
) -> dict[str, Any]:
    rows = []
    for _ in range(instances):
        L = structured_kernel(n_groups, group_size, rng, **kernel_kwargs)
        result = solve_step(L, n_groups, group_size)
        row = {
            "single_eq_multi": int(abs(result["single"] - result["multi"]) <= TIE_NATS),
            "degenerate": int(result["opt"] - result["min"] <= TIE_NATS),
            "mask_frac": 0.5,
        }
        row.update(gap_row(result, result["single"], result["single_set"], "single"))
        row.update(gap_row(result, result["multi"], result["multi_set"], "multi"))
        row["prod_delta"] = row["multi_delta"]
        row["fnw_holds"] = int(result["single"] >= fnw_bound(L, result["opt"], n_groups) - TIE_NATS)
        rows.append(row)
    live = [r for r in rows if not r["degenerate"]]
    single = np.array([r["single_delta"] for r in live])
    multi = np.array([r["multi_delta"] for r in live])
    return {
        "shape": f"{n_groups}x{group_size}",
        "instances": len(live),
        "single_optimal": float(np.mean([r["single_optimal"] for r in live])),
        "multi_optimal": float(np.mean([r["multi_optimal"] for r in live])),
        "single_eq_multi": float(np.mean([r["single_eq_multi"] for r in live])),
        "single_delta_mean": float(single.mean()),
        "single_delta_max": float(single.max()),
        "single_det_ratio_min": float(np.exp(-single.max())),
        "multi_delta_max": float(multi.max()),
        "single_norm_gap_mean": float(np.mean([r["single_norm_gap"] for r in live])),
        "fnw_holds": float(np.mean([r["fnw_holds"] for r in live])),
        **kernel_kwargs,
    }


def cmd_theory(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {}

    report["h1_equicorrelated"] = equicorrelated_rows((0.3, 0.5, 0.7, 0.9, 0.95, 0.99), (3, 4, 8))
    print("H.1, equicorrelated C = c (11^T - I):")
    print(
        f"  {'k':>2} {'cos':>5} {'||C||2':>7} {'det(I+C)':>10} {'order 2':>9} {'order 3':>9} {'H.1 bound':>9} "
        f"{'Fischer':>8}",
    )
    for r in report["h1_equicorrelated"]:
        print(
            f"  {r['k']:>2} {r['cos']:>5.2f} {r['C_norm']:>7.2f} {r['det_exact']:>10.2e} {r['det_order2']:>9.2f} "
            f"{r['det_order3']:>9.2f} {r['cs_bound']:>9.3f} {r['fischer_bound']:>8.3f}",
        )

    random_rows = []
    for k in (3, 4, 8):
        for mean_cos in (0.5, 0.9, 0.99):
            c_norms, h_norms, rhos, dets, cos = [], [], [], [], []
            cs_violated, fischer_violated = [], []
            for _ in range(200):
                x = unit_vectors_with_mean_cosine(k, mean_cos, 4096, rng)
                S = x @ x.T
                c_norm, h_norm, rho = expansion_norms(S)
                off = S[np.triu_indices(k, 1)]
                one_minus_det = max(1 - np.linalg.det(S), 0.0)
                cs_violated.append(off.mean() > np.sqrt(one_minus_det / len(off)))
                fischer_violated.append(np.abs(off).max() > np.sqrt(one_minus_det) + 1e-12)
                c_norms.append(c_norm)
                h_norms.append(h_norm)
                rhos.append(rho)
                dets.append(np.linalg.det(S))
                cos.append(S[np.triu_indices(k, 1)].mean())
            random_rows.append(
                {
                    "k": k,
                    "mean_cos": float(np.mean(cos)),
                    "C_norm_mean": float(np.mean(c_norms)),
                    "det_mean": float(np.mean(dets)),
                    "order2_mean": float(1 - k * (k - 1) / 2 * np.mean(cos) ** 2),
                    "H_norm_min_rho_mean": float(np.mean(h_norms)),
                    "rho_mean": float(np.mean(rhos)),
                    "h1_bound_violated_frac": float(np.mean(cs_violated)),
                    "fischer_bound_violated_frac": float(np.mean(fischer_violated)),
                },
            )
    report["h_random_shared_component"] = random_rows
    print("\nShared component + isotropic residual (dim 4096), 200 draws:")
    for r in random_rows:
        print(
            f"  k={r['k']} mean cos {r['mean_cos']:.3f}: ||C||2 {r['C_norm_mean']:.2f}, det {r['det_mean']:.2e}, "
            f"order-2 approx {r['order2_mean']:.2f}, min_rho ||H||2 {r['H_norm_min_rho_mean']:.3f} "
            f"(rho {r['rho_mean']:.2f}); H.1 cosine bound violated {r['h1_bound_violated_frac']:.2f}, "
            f"Fischer bound violated {r['fischer_bound_violated_frac']:.2f}",
        )

    solver = []
    settings = [
        {"lineage_cos": 0.9, "sibling_cos": None, "beta": 20.0},
        {"lineage_cos": 0.9, "sibling_cos": 0.99, "beta": 20.0},
        {"lineage_cos": 0.99, "sibling_cos": 0.999, "beta": 20.0},
        {"lineage_cos": 0.5, "sibling_cos": 0.9, "beta": 20.0},
        {"lineage_cos": 0.9, "sibling_cos": 0.99, "beta": 1.0},
        {"lineage_cos": 0.9, "sibling_cos": 0.99, "beta": 0.05},
    ]
    for setting in settings:
        for k, w in ((3, 3), (4, 4), (8, 4)):
            n = args.instances if k * w <= 16 else max(args.instances // 5, 50)
            solver.append(synthetic_solver(k, w, n, rng, **setting))
    report["synthetic_solver"] = solver
    print("\nExact vs greedy transversal MAP on structured kernels (L = diag(Q) + beta S):")
    for r in solver:
        print(
            f"  {r['shape']:>4} beta {r['beta']:>5g} cos {r['lineage_cos']}/{r['sibling_cos']}: n={r['instances']}, "
            f"single optimal {r['single_optimal']:.3f}, multi optimal {r['multi_optimal']:.3f}, "
            f"single==multi {r['single_eq_multi']:.3f}, worst det(G)/det(OPT) single {r['single_det_ratio_min']:.4f}, "
            f"mean norm gap {r['single_norm_gap_mean']:.4f}, FNW bound holds {r['fnw_holds']:.2f}",
        )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "theory.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m d5p4.selection_dump_analysis", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("solver")
    p.add_argument("dump_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("proxy")
    p.add_argument("dump_dir", type=Path)
    p.add_argument("--results", type=Path, required=True, help="Generation JSON of the same run.")
    p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("theory")
    p.add_argument("--instances", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    return {"solver": cmd_solver, "proxy": cmd_proxy, "theory": cmd_theory}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

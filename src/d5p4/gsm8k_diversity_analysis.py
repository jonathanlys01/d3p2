"""Analyze whether intra-prompt diversity is associated with GSM8K recovery.

The input files are never modified. All caches, tables, figures, and reports
are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from d5p4.eval_utils import _compute_self_bleu_impl, _get_sentence_bleu_metric
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.text_postprocessors import MathParser
from d5p4.utils import process_model_args


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ANALYSIS_VERSION = "gsm8k-diversity-v2"
LEXICAL_CACHE_VERSION = "gsm8k-diversity-v1"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("_default.yaml")
METHOD_LABELS = {
    "baseline": "Standard sampling",
    "greedy_map": "D5P4",
    "diverse_beam": "Diverse beam",
    "greedy_beam": "Greedy beam",
}
BUCKET_ORDER = ("Easy", "Hard / Recovered", "Unsolved")
PRIMARY_METRICS = ("lexical_diversity", "semantic_diversity")
ROBUSTNESS_METRICS = (
    "rationale_lexical_diversity",
    "rationale_semantic_diversity",
    "incorrect_pairwise_lexical_distance",
    "incorrect_semantic_diversity",
)
MATCH_CONFIG_KEYS = (
    "model",
    "qa_dataset",
    "qa_dataset_len",
    "qa_n_shots",
    "cfg_scale",
    "llada_steps",
    "dream_steps",
    "gen_length",
    "block_length",
    "remasking",
    "selection_temperature",
    "cat_temperature",
    "logits_eos_inf",
    "confidence_eos_eot_inf",
    "guidance_start",
    "guidance_end",
)
EXCLUDED_NAME_MARKERS = ("-math-bon-", "-metrics", "temp")
_ANSWER_CUE_RE = re.compile(r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)")
_NUMBER_RE = re.compile(r"-?(?:\d+(?:,\d{3})*|\.\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?")


@dataclass(frozen=True)
class PromptResult:
    prompt_id: str
    question: str
    gold_answer: str
    generations: tuple[str, ...]
    scores: tuple[int, ...]


@dataclass(frozen=True)
class MathRun:
    path: Path
    family: str
    seed: str
    method: str
    method_label: str
    config: dict[str, Any]
    prompts: dict[str, PromptResult]
    k: int


@dataclass(frozen=True)
class LexicalMetrics:
    self_bleu: float
    lexical_diversity: float
    pairwise_lexical_distance: float
    unique_fraction: float


@dataclass(frozen=True)
class AnalysisDefaults:
    cos_model_id: str
    model_cache_dir: Path


@dataclass(frozen=True)
class CandidateSelection:
    target_k: int
    grouped_methods: tuple[str, ...]


def _available_candidate_count(run: MathRun, baseline_method: str) -> int:
    group_size = int(run.config.get("group_size", 1))
    if run.method == baseline_method:
        if group_size != 1:
            raise ValueError(f"{run.path}: the pass@1 baseline must have group_size=1")
        return run.k - 1
    return run.k // group_size if group_size > 1 else run.k


def _stable_hash(*parts: str) -> str:
    payload = "\x1f".join(parts).encode()
    return hashlib.sha256(payload).hexdigest()


def load_analysis_defaults(config_path: Path) -> AnalysisDefaults:
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    config = OmegaConf.load(path)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    cos_model_id = resolved.get("cos_model_id")
    cache_dir = resolved.get("cache_dir")
    if not isinstance(cos_model_id, str) or not cos_model_id:
        raise ValueError(f"config is missing cos_model_id: {path}")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError(f"config is missing cache_dir: {path}")
    model_cache_dir = Path(cache_dir).expanduser()
    if not model_cache_dir.is_absolute():
        model_cache_dir = (Path.cwd() / model_cache_dir).resolve()
    return AnalysisDefaults(cos_model_id=cos_model_id, model_cache_dir=model_cache_dir)


def prompt_id(question: str, gold_answer: str) -> str:
    return _stable_hash(question.strip(), gold_answer.strip())


def _comparison_family(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    if len(parts) < 3:
        return "."
    parent = Path(*parts[:-2])
    return str(parent) if str(parent) else "."


def _extract_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    results = data.get("results")
    if isinstance(results, dict):
        results = results.get("results")
    if not isinstance(results, list):
        raise ValueError("missing list-valued results")
    if not results:
        raise ValueError("results is empty")
    if not all(isinstance(row, dict) for row in results):
        raise ValueError("every result row must be an object")
    return results


def _load_run(root: Path, path: Path) -> MathRun | None:  # noqa: C901, PLR0912
    if any(marker in path.name for marker in EXCLUDED_NAME_MARKERS):
        return None
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        return None
    config = dict(data["config"])
    if config.get("qa_dataset") != "gsm8k":
        return None
    method = config.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"{path}: config.method must be a non-empty string")
    try:
        rows = _extract_results(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    prompts: dict[str, PromptResult] = {}
    candidate_counts: set[int] = set()
    for row_idx, row in enumerate(rows):
        question = row.get("question")
        gold_answer = row.get("gold_answer")
        generations = row.get("generations")
        scores = row.get("scores")
        if not isinstance(question, str) or not isinstance(generations, list) or not isinstance(scores, list):
            raise ValueError(f"{path}: result {row_idx} lacks question/generations/scores")
        if gold_answer is None:
            raise ValueError(f"{path}: result {row_idx} lacks gold_answer")
        if not generations or len(generations) != len(scores):
            raise ValueError(f"{path}: result {row_idx} generation/score lengths do not match")
        if not all(isinstance(text, str) for text in generations):
            raise ValueError(f"{path}: result {row_idx} contains a non-string generation")
        if not all(score in (0, 1, False, True) for score in scores):
            raise ValueError(f"{path}: result {row_idx} scores must be binary")
        gold = str(gold_answer)
        key = prompt_id(question, gold)
        if key in prompts:
            raise ValueError(f"{path}: duplicate question/gold prompt at result {row_idx}")
        prompts[key] = PromptResult(
            prompt_id=key,
            question=question,
            gold_answer=gold,
            generations=tuple(generations),
            scores=tuple(int(score) for score in scores),
        )
        candidate_counts.add(len(generations))
    if len(candidate_counts) != 1:
        raise ValueError(f"{path}: inconsistent candidate counts {sorted(candidate_counts)}")

    return MathRun(
        path=path,
        family=_comparison_family(root, path),
        seed=str(config.get("seed", "unspecified")),
        method=method,
        method_label=METHOD_LABELS.get(method, method.replace("_", " ").title()),
        config=config,
        prompts=prompts,
        k=candidate_counts.pop(),
    )


def discover_runs(  # noqa: C901, PLR0912
    results_root: Path,
    baseline_method: str = "baseline",
) -> list[MathRun]:
    root = results_root.resolve()
    if not root.is_dir():
        raise ValueError(f"results root is not a directory: {root}")
    runs = [run for path in sorted(root.rglob("*.json")) if (run := _load_run(root, path)) is not None]
    if not runs:
        raise ValueError(f"no raw GSM8K result JSONs found below {root}")

    grouped: dict[tuple[str, str], list[MathRun]] = {}
    seen: dict[tuple[str, str, str], Path] = {}
    for run in runs:
        unique_key = (run.family, run.seed, run.method)
        if unique_key in seen:
            raise ValueError(
                f"ambiguous duplicate for family={run.family!r}, seed={run.seed!r}, method={run.method!r}: "
                f"{seen[unique_key]} and {run.path}",
            )
        seen[unique_key] = run.path
        grouped.setdefault((run.family, run.seed), []).append(run)

    for (family, seed), group_runs in grouped.items():
        baselines = [run for run in group_runs if run.method == baseline_method]
        if len(baselines) != 1:
            raise ValueError(f"family={family!r}, seed={seed!r} requires exactly one {baseline_method!r} run")
        baseline = baselines[0]
        baseline_ids = set(baseline.prompts)
        for run in group_runs:
            if set(run.prompts) != baseline_ids:
                missing = len(baseline_ids - set(run.prompts))
                extra = len(set(run.prompts) - baseline_ids)
                raise ValueError(
                    f"{run.path}: prompt set differs from baseline (missing={missing}, extra={extra})",
                )
            if run.k != baseline.k:
                raise ValueError(f"{run.path}: k={run.k}, but baseline k={baseline.k}")
            for key in MATCH_CONFIG_KEYS:
                if key in baseline.config and key in run.config and baseline.config[key] != run.config[key]:
                    raise ValueError(
                        f"{run.path}: comparison control {key!r}={run.config[key]!r} "
                        f"does not match baseline value {baseline.config[key]!r}",
                    )
            for item_id, baseline_prompt in baseline.prompts.items():
                prompt = run.prompts[item_id]
                if prompt.question != baseline_prompt.question or prompt.gold_answer != baseline_prompt.gold_answer:
                    raise ValueError(f"{run.path}: prompt hash collision for {item_id}")
    return runs


def validation_summary(runs: Sequence[MathRun]) -> dict[str, Any]:
    groups = {(run.family, run.seed) for run in runs}
    return {
        "analysis_version": ANALYSIS_VERSION,
        "files": len(runs),
        "families": len({run.family for run in runs}),
        "replicates": len(groups),
        "seeds": sorted({run.seed for run in runs}),
        "methods": sorted({run.method for run in runs}),
        "prompt_counts": sorted({len(run.prompts) for run in runs}),
        "candidate_counts": sorted({run.k for run in runs}),
        "runs": [
            {
                "path": str(run.path),
                "family": run.family,
                "seed": run.seed,
                "method": run.method,
                "prompts": len(run.prompts),
                "k": run.k,
            }
            for run in runs
        ],
    }


def candidate_selection_layout(
    runs: Sequence[MathRun],
    baseline_method: str = "baseline",
) -> CandidateSelection:
    grouped_counts: dict[tuple[str, str, str], int] = {}
    for run in runs:
        group_size = run.config.get("group_size", 1)
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError(f"{run.path}: config.group_size must be a positive integer")
        if run.k % group_size:
            raise ValueError(f"{run.path}: raw k={run.k} is not divisible by group_size={group_size}")
        if group_size > 1:
            grouped_counts[(run.family, run.seed, run.method)] = run.k // group_size

    target_counts = set(grouped_counts.values())
    if len(target_counts) > 1:
        details = ", ".join(
            f"{family}/{seed}/{method}={count}"
            for (family, seed, method), count in sorted(grouped_counts.items())
        )
        raise ValueError(f"grouped methods produce different final candidate counts: {details}")
    target_k = (
        target_counts.pop()
        if target_counts
        else min(_available_candidate_count(run, baseline_method) for run in runs)
    )
    if target_k < 1:
        raise ValueError("candidate selection requires at least one candidate beyond the pass@1 anchor")
    for run in runs:
        available = _available_candidate_count(run, baseline_method)
        if available < target_k:
            raise ValueError(
                f"{run.path}: only {available} final candidates are available, but target k={target_k}",
            )
    grouped_methods = tuple(sorted({method for _, _, method in grouped_counts}))
    return CandidateSelection(target_k=target_k, grouped_methods=grouped_methods)


def select_candidate_indices(
    run: MathRun,
    result: PromptResult,
    target_k: int,
    selection_seed: int,
    baseline_method: str = "baseline",
) -> tuple[int, ...]:
    group_size = int(run.config.get("group_size", 1))
    seed_digest = _stable_hash(
        str(selection_seed),
        run.family,
        run.seed,
        run.method,
        result.prompt_id,
    )
    rng = np.random.default_rng(int(seed_digest[:16], 16))
    if group_size > 1:
        group_starts = range(0, run.k, group_size)
        indices = tuple(start + int(rng.integers(group_size)) for start in group_starts)
        if len(indices) != target_k:
            raise ValueError(
                f"{run.path}: grouped selection produced {len(indices)} candidates, expected {target_k}",
            )
        return indices
    start = 1 if run.method == baseline_method else 0
    selected = rng.choice(np.arange(start, run.k), size=target_k, replace=False)
    return tuple(sorted(int(index) for index in selected))


def final_answer_controlled(text: str, parser: MathParser | None = None) -> tuple[str, bool]:
    """Remove or mask a terminal answer span while preserving the rationale."""
    stripped = text.strip()
    marker = stripped.rfind("####")
    if marker >= 0:
        rationale = stripped[:marker].rstrip()
        return (rationale or "<EMPTY_RATIONALE>", True)

    boxed_matches = list(MathParser._BOXED_RE.finditer(stripped))  # noqa: SLF001
    if boxed_matches:
        match = boxed_matches[-1]
        controlled = (stripped[: match.start()] + "<FINAL_ANSWER>" + stripped[match.end() :]).strip()
        return controlled, True

    lines = stripped.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        cue = _ANSWER_CUE_RE.search(lines[idx])
        if cue:
            controlled_lines = [*lines[:idx], lines[idx][: cue.start()]]
            controlled = "\n".join(controlled_lines).strip()
            return (controlled or "<EMPTY_RATIONALE>", True)

    numeric_parser = parser or MathParser()
    extracted = numeric_parser.extract_universal_numeric(stripped)
    if extracted == "NULL":
        return stripped, False
    candidates = list(_NUMBER_RE.finditer(stripped))
    for match in reversed(candidates):
        normalized = numeric_parser.extract_universal_numeric(match.group())
        if normalized == extracted:
            controlled = (stripped[: match.start()] + "<FINAL_ANSWER>" + stripped[match.end() :]).strip()
            return controlled, True
    return stripped, False


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return float("nan")
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)
    distances = 1.0 - matrix @ matrix.T
    upper = distances[np.triu_indices(len(matrix), k=1)]
    return float(np.mean(upper))


def _mean_pairwise_bleu_distance(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return float("nan")
    bleu = _get_sentence_bleu_metric()
    distances: list[float] = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            forward = bleu.sentence_score(texts[left], [texts[right]]).score
            backward = bleu.sentence_score(texts[right], [texts[left]]).score
            distances.append(1.0 - ((forward + backward) / 200.0))
    return float(np.mean(distances))


def compute_lexical_metrics(texts: Sequence[str], include_pairwise: bool = False) -> LexicalMetrics:
    if len(texts) < 2:
        return LexicalMetrics(float("nan"), float("nan"), float("nan"), float("nan"))
    self_bleu = float(_compute_self_bleu_impl(list(texts)))
    return LexicalMetrics(
        self_bleu=self_bleu,
        lexical_diversity=1.0 - self_bleu / 100.0,
        pairwise_lexical_distance=_mean_pairwise_bleu_distance(texts) if include_pairwise else float("nan"),
        unique_fraction=len(set(texts)) / len(texts),
    )


def _lexical_worker(task: tuple[str, tuple[str, ...], bool]) -> tuple[str, LexicalMetrics]:
    group_hash, texts, include_pairwise = task
    return group_hash, compute_lexical_metrics(texts, include_pairwise=include_pairwise)


class AnalysisCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_metrics (
                version TEXT NOT NULL,
                group_hash TEXT NOT NULL,
                self_bleu REAL,
                lexical_diversity REAL,
                pairwise_lexical_distance REAL,
                unique_fraction REAL,
                PRIMARY KEY (version, group_hash)
            )
            """,
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_fingerprint TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model_fingerprint, text_hash)
            )
            """,
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_lexical(self, group_hash: str) -> LexicalMetrics | None:
        row = self.connection.execute(
            """
            SELECT self_bleu, lexical_diversity, pairwise_lexical_distance, unique_fraction
            FROM lexical_metrics WHERE version = ? AND group_hash = ?
            """,
            (LEXICAL_CACHE_VERSION, group_hash),
        ).fetchone()
        if row is None:
            return None
        values = (float(value) if value is not None else float("nan") for value in row)
        return LexicalMetrics(*values)

    def put_lexical(self, group_hash: str, metrics: LexicalMetrics) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO lexical_metrics VALUES (?, ?, ?, ?, ?, ?)",
            (
                LEXICAL_CACHE_VERSION,
                group_hash,
                metrics.self_bleu,
                metrics.lexical_diversity,
                metrics.pairwise_lexical_distance,
                metrics.unique_fraction,
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def get_embeddings(self, model_fingerprint: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        hashes = {_stable_hash(text): text for text in texts}
        hash_items = list(hashes)
        for start in range(0, len(hash_items), 500):
            chunk = hash_items[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT text_hash, dimension, vector FROM embeddings
                WHERE model_fingerprint = ? AND text_hash IN ({placeholders})
                """,  # noqa: S608
                (model_fingerprint, *chunk),
            ).fetchall()
            for text_hash, dimension, vector in rows:
                output[hashes[text_hash]] = np.frombuffer(vector, dtype=np.float32, count=dimension).copy()
        return output

    def put_embeddings(
        self,
        model_fingerprint: str,
        texts: Sequence[str],
        vectors: np.ndarray,
    ) -> None:
        rows = []
        for text, vector in zip(texts, vectors):
            array = np.asarray(vector, dtype=np.float32)
            rows.append((model_fingerprint, _stable_hash(text), text, array.size, array.tobytes()))
        self.connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)", rows)
        self.connection.commit()


def _group_hash(texts: Sequence[str], include_pairwise: bool) -> str:
    return _stable_hash(LEXICAL_CACHE_VERSION, "pairwise" if include_pairwise else "self", *texts)


def compute_lexical_groups(
    groups: dict[str, tuple[tuple[str, ...], bool]],
    cache: AnalysisCache,
    num_workers: int,
) -> dict[str, LexicalMetrics]:
    output: dict[str, LexicalMetrics] = {}
    missing: list[tuple[str, tuple[str, ...], bool]] = []
    names_by_hash: dict[str, list[str]] = {}
    for name, (texts, include_pairwise) in groups.items():
        group_hash = _group_hash(texts, include_pairwise)
        names_by_hash.setdefault(group_hash, []).append(name)
        cached = cache.get_lexical(group_hash)
        if cached is None:
            if len(names_by_hash[group_hash]) == 1:
                missing.append((group_hash, texts, include_pairwise))
        else:
            output[name] = cached

    def record(group_hash: str, metrics: LexicalMetrics) -> None:
        cache.put_lexical(group_hash, metrics)
        for name in names_by_hash[group_hash]:
            output[name] = metrics

    if num_workers > 1 and missing:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for group_hash, metrics in pool.map(_lexical_worker, missing, chunksize=8):
                record(group_hash, metrics)
    else:
        for task in missing:
            group_hash, metrics = _lexical_worker(task)
            record(group_hash, metrics)
    cache.commit()
    return output


def model_fingerprint(model_id: str) -> str:
    path = Path(model_id).expanduser()
    config_digest = ""
    if path.is_dir() and (path / "config.json").is_file():
        config_digest = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    return _stable_hash(str(path.resolve()) if path.exists() else model_id, config_digest)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def load_embeddings(  # noqa: PLR0913
    texts: Sequence[str],
    cache: AnalysisCache,
    model_id: str,
    requested_device: str,
    batch_size: int,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], str]:
    unique_texts = sorted(set(texts))
    fingerprint = model_fingerprint(model_id)
    embeddings = cache.get_embeddings(fingerprint, unique_texts)
    missing = [text for text in unique_texts if text not in embeddings]
    if missing:
        args = process_model_args(model_id, cache_dir=str(cache_dir))
        model = JinaBertModel.from_pretrained(**args)
        device = resolve_device(requested_device)
        model.to(device)
        model.eval()
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            encoded = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=device,
                normalize_embeddings=True,
            )
            vectors = np.asarray(encoded, dtype=np.float32)
            cache.put_embeddings(fingerprint, batch, vectors)
            embeddings.update(dict(zip(batch, vectors)))
    return embeddings, fingerprint


def _baseline_lookup(runs: Sequence[MathRun], baseline_method: str) -> dict[tuple[str, str], MathRun]:
    return {(run.family, run.seed): run for run in runs if run.method == baseline_method}


def classify_bucket(baseline_pass1: int, observed_passk: int) -> str:
    if baseline_pass1:
        return "Easy"
    if observed_passk:
        return "Hard / Recovered"
    return "Unsolved"


def build_prompt_rows(  # noqa: PLR0913
    runs: Sequence[MathRun],
    cache: AnalysisCache,
    cos_model_id: str,
    requested_device: str,
    batch_size: int,
    num_workers: int,
    model_cache_dir: Path,
    selection_seed: int,
    baseline_method: str = "baseline",
) -> tuple[pd.DataFrame, str]:
    parser = MathParser()
    baselines = _baseline_lookup(runs, baseline_method)
    selection = candidate_selection_layout(runs, baseline_method)
    lexical_groups: dict[str, tuple[tuple[str, ...], bool]] = {}
    prepared: list[dict[str, Any]] = []
    all_embedding_texts: list[str] = []

    for run in runs:
        baseline = baselines[(run.family, run.seed)]
        for item_id, result in run.prompts.items():
            baseline_pass1 = baseline.prompts[item_id].scores[0]
            selected_indices = select_candidate_indices(
                run,
                result,
                selection.target_k,
                selection_seed,
                baseline_method,
            )
            selected_generations = tuple(result.generations[index] for index in selected_indices)
            selected_scores = tuple(result.scores[index] for index in selected_indices)
            observed_passk = int(any(selected_scores))
            bucket = classify_bucket(baseline_pass1, observed_passk)

            rationale_pairs = [final_answer_controlled(text, parser) for text in selected_generations]
            rationale_texts = tuple(text for text, _ in rationale_pairs)
            incorrect_texts = tuple(
                text for text, score in zip(selected_generations, selected_scores) if score == 0
            )
            base_key = _stable_hash(str(run.path), item_id)
            lexical_groups[f"{base_key}:full"] = (selected_generations, False)
            lexical_groups[f"{base_key}:rationale"] = (rationale_texts, False)
            if len(incorrect_texts) >= 2:
                lexical_groups[f"{base_key}:incorrect"] = (incorrect_texts, True)
            all_embedding_texts.extend(selected_generations)
            all_embedding_texts.extend(rationale_texts)
            prepared.append(
                {
                    "base_key": base_key,
                    "run": run,
                    "result": result,
                    "selected_indices": selected_indices,
                    "selected_generations": selected_generations,
                    "selected_scores": selected_scores,
                    "rationale_texts": rationale_texts,
                    "rationale_mask_coverage": sum(masked for _, masked in rationale_pairs) / selection.target_k,
                    "incorrect_texts": incorrect_texts,
                    "baseline_pass1": baseline_pass1,
                    "observed_passk": observed_passk,
                    "bucket": bucket,
                },
            )

    lexical = compute_lexical_groups(lexical_groups, cache, num_workers)
    embeddings, fingerprint = load_embeddings(
        all_embedding_texts,
        cache,
        cos_model_id,
        requested_device,
        batch_size,
        model_cache_dir,
    )

    rows: list[dict[str, Any]] = []
    for item in prepared:
        run: MathRun = item["run"]
        result: PromptResult = item["result"]
        selected_generations: tuple[str, ...] = item["selected_generations"]
        selected_scores: tuple[int, ...] = item["selected_scores"]
        base_key = item["base_key"]
        full_lexical = lexical[f"{base_key}:full"]
        rationale_lexical = lexical[f"{base_key}:rationale"]
        incorrect_texts = item["incorrect_texts"]
        incorrect_lexical = lexical.get(f"{base_key}:incorrect")
        full_vectors = np.stack([embeddings[text] for text in selected_generations])
        rationale_vectors = np.stack([embeddings[text] for text in item["rationale_texts"]])
        incorrect_vectors = (
            np.stack([embeddings[text] for text in incorrect_texts]) if len(incorrect_texts) >= 2 else None
        )
        rows.append(
            {
                "family": run.family,
                "seed": run.seed,
                "method": run.method,
                "method_label": run.method_label,
                "source_path": str(run.path),
                "prompt_id": result.prompt_id,
                "question": result.question,
                "gold_answer": result.gold_answer,
                "raw_k": run.k,
                "k": selection.target_k,
                "group_size": int(run.config.get("group_size", 1)),
                "candidate_selection": (
                    "random_excluding_pass1_anchor"
                    if run.method == baseline_method
                    else (
                        "random_one_per_group"
                        if run.config.get("group_size", 1) > 1
                        else "random"
                    )
                ),
                "selected_indices": json.dumps(item["selected_indices"]),
                "baseline_pass1": item["baseline_pass1"],
                "observed_passk": item["observed_passk"],
                "marginal_gain": item["observed_passk"] - item["baseline_pass1"],
                "bucket": item["bucket"],
                "n_correct": sum(selected_scores),
                "unique_fraction": full_lexical.unique_fraction,
                "self_bleu": full_lexical.self_bleu,
                "lexical_diversity": full_lexical.lexical_diversity,
                "semantic_diversity": mean_pairwise_cosine_distance(full_vectors),
                "rationale_mask_coverage": item["rationale_mask_coverage"],
                "rationale_self_bleu": rationale_lexical.self_bleu,
                "rationale_lexical_diversity": rationale_lexical.lexical_diversity,
                "rationale_semantic_diversity": mean_pairwise_cosine_distance(rationale_vectors),
                "incorrect_n": len(incorrect_texts),
                "incorrect_lexical_diversity": (
                    incorrect_lexical.lexical_diversity if incorrect_lexical is not None else float("nan")
                ),
                "incorrect_pairwise_lexical_distance": (
                    incorrect_lexical.pairwise_lexical_distance
                    if incorrect_lexical is not None
                    else float("nan")
                ),
                "incorrect_semantic_diversity": (
                    mean_pairwise_cosine_distance(incorrect_vectors)
                    if incorrect_vectors is not None
                    else float("nan")
                ),
            },
        )
    return pd.DataFrame(rows), fingerprint


def _difference_statistic(values: np.ndarray, outcomes: np.ndarray) -> float:
    recovered = values[outcomes == 1]
    unsolved = values[outcomes == 0]
    if len(recovered) == 0 or len(unsolved) == 0:
        return float("nan")
    return float(np.nanmean(recovered) - np.nanmean(unsolved))


def clustered_difference_test(  # noqa: PLR0913
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    outcome_column: str,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    subset = frame[[cluster_column, value_column, outcome_column]].dropna()
    cluster_series = cast(pd.Series, subset[cluster_column])
    clusters = sorted(cluster_series.unique())
    if not clusters:
        return {
            "effect": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    grouped = [
        cast(pd.DataFrame, subset.loc[cluster_series == cluster])
        for cluster in clusters
    ]
    values = [cast(pd.Series, group[value_column]).to_numpy(float) for group in grouped]
    outcomes = [cast(pd.Series, group[outcome_column]).to_numpy(int) for group in grouped]
    flat_values = np.concatenate(values)
    flat_outcomes = np.concatenate(outcomes)
    observed = _difference_statistic(flat_values, flat_outcomes)
    if not np.isfinite(observed):
        return {"effect": observed, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    bootstrap = np.empty(bootstrap_reps, dtype=float)
    for rep in range(bootstrap_reps):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap[rep] = _difference_statistic(
            np.concatenate([values[idx] for idx in sampled]),
            np.concatenate([outcomes[idx] for idx in sampled]),
        )
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    ci_low, ci_high = (
        np.quantile(finite_bootstrap, [0.025, 0.975]) if len(finite_bootstrap) else (float("nan"), float("nan"))
    )

    permutations = np.empty(permutation_reps, dtype=float)
    indices_by_size: dict[int, list[int]] = {}
    for cluster_idx, cluster_outcome in enumerate(outcomes):
        indices_by_size.setdefault(len(cluster_outcome), []).append(cluster_idx)
    for rep in range(permutation_reps):
        permuted: list[np.ndarray | None] = [None] * len(clusters)
        for same_size_indices in indices_by_size.values():
            sources = rng.permutation(same_size_indices)
            for target, source in zip(same_size_indices, sources):
                permuted[target] = outcomes[source]
        assert all(item is not None for item in permuted)
        permutations[rep] = _difference_statistic(
            flat_values,
            np.concatenate(cast(list[np.ndarray], permuted)),
        )
    p_value = (1 + int(np.sum(np.abs(permutations) >= abs(observed)))) / (permutation_reps + 1)
    return {
        "effect": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }


def clustered_paired_test(  # noqa: PLR0913
    differences: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    subset = differences[[cluster_column, value_column]].dropna()
    cluster_values = subset.groupby(cluster_column)[value_column].mean().to_numpy(float)
    if len(cluster_values) == 0:
        return {"effect": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    observed = float(np.mean(cluster_values))
    sampled = rng.integers(0, len(cluster_values), size=(bootstrap_reps, len(cluster_values)))
    boot = np.mean(cluster_values[sampled], axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    signs = rng.choice((-1.0, 1.0), size=(permutation_reps, len(cluster_values)))
    permuted = np.mean(signs * cluster_values, axis=1)
    p_value = (1 + int(np.sum(np.abs(permuted) >= abs(observed)))) / (permutation_reps + 1)
    return {
        "effect": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted.tolist()
    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order] * len(finite_values) / np.arange(1, len(finite_values) + 1)
    corrected_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty_like(corrected_sorted)
    corrected[order] = corrected_sorted
    adjusted[finite] = np.clip(corrected, 0.0, 1.0)
    return adjusted.tolist()


def run_inference(
    frame: pd.DataFrame,
    bootstrap_reps: int,
    permutation_reps: int,
    analysis_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(analysis_seed)
    tests: list[dict[str, Any]] = []
    hard = cast(pd.DataFrame, frame.loc[cast(pd.Series, frame["baseline_pass1"]) == 0])
    for method_label, method_frame in hard.groupby("method_label", sort=True):
        for metric in (*PRIMARY_METRICS, *ROBUSTNESS_METRICS):
            result = clustered_difference_test(
                method_frame,
                metric,
                "prompt_id",
                "observed_passk",
                bootstrap_reps,
                permutation_reps,
                rng,
            )
            tests.append(
                {
                    "analysis_type": "recovered_minus_unsolved",
                    "method_a": method_label,
                    "method_b": "",
                    "metric": metric,
                    "n": int(cast(pd.Series, method_frame[metric]).notna().sum()),
                    **result,
                },
            )

    methods = sorted(cast(pd.Series, hard["method_label"]).unique())
    pair_metrics = ("observed_passk", *PRIMARY_METRICS)
    key_columns = ["family", "seed", "prompt_id"]
    for left_idx, left in enumerate(methods):
        for right in methods[left_idx + 1 :]:
            method_series = cast(pd.Series, hard["method_label"])
            left_frame = cast(
                pd.DataFrame,
                hard.loc[method_series == left, key_columns + list(pair_metrics)],
            )
            right_frame = cast(
                pd.DataFrame,
                hard.loc[method_series == right, key_columns + list(pair_metrics)],
            )
            paired = left_frame.merge(right_frame, on=key_columns, suffixes=("_a", "_b"), validate="one_to_one")
            for metric in pair_metrics:
                difference_column = f"{metric}_difference"
                paired[difference_column] = paired[f"{metric}_a"] - paired[f"{metric}_b"]
                result = clustered_paired_test(
                    paired,
                    difference_column,
                    "prompt_id",
                    bootstrap_reps,
                    permutation_reps,
                    rng,
                )
                tests.append(
                    {
                        "analysis_type": "paired_method_contrast",
                        "method_a": left,
                        "method_b": right,
                        "metric": metric,
                        "n": int(cast(pd.Series, paired[difference_column]).notna().sum()),
                        **result,
                    },
                )
    output = pd.DataFrame(tests)
    output["p_adjusted_bh"] = benjamini_hochberg(output["p_value"].tolist())
    return output


def build_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "observed_passk",
        "marginal_gain",
        "lexical_diversity",
        "semantic_diversity",
        "unique_fraction",
        "rationale_lexical_diversity",
        "rationale_semantic_diversity",
        "incorrect_pairwise_lexical_distance",
        "incorrect_semantic_diversity",
    ]
    summary = (
        frame.groupby(["family", "method_label", "bucket"], observed=True)[metrics]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    hard = cast(pd.DataFrame, frame.loc[cast(pd.Series, frame["baseline_pass1"]) == 0])
    fixed_hard = hard.groupby("method_label").agg(
        hard_n=("prompt_id", "size"),
        recovery_rate=("observed_passk", "mean"),
        lexical_diversity=("lexical_diversity", "mean"),
        semantic_diversity=("semantic_diversity", "mean"),
        unique_fraction=("unique_fraction", "mean"),
    )
    recovered_frame = cast(
        pd.DataFrame,
        hard.loc[cast(pd.Series, hard["observed_passk"]) == 1],
    )
    recovered = recovered_frame.groupby("method_label").agg(
        recovered_n=("prompt_id", "size"),
        recovered_lexical_diversity=("lexical_diversity", "mean"),
        recovered_semantic_diversity=("semantic_diversity", "mean"),
    )
    matrix = fixed_hard.join(recovered, how="left").reset_index()
    return summary, matrix


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _bucket_positions(methods: Sequence[str]) -> tuple[list[float], list[str], dict[tuple[str, str], float]]:
    positions: list[float] = []
    labels: list[str] = []
    lookup: dict[tuple[str, str], float] = {}
    position = 1.0
    for method in methods:
        for bucket in BUCKET_ORDER:
            positions.append(position)
            labels.append(f"{method}\n{bucket}")
            lookup[(method, bucket)] = position
            position += 1.0
        position += 0.6
    return positions, labels, lookup


def plot_bucket_distributions(frame: pd.DataFrame, output_dir: Path) -> None:
    methods = sorted(cast(pd.Series, frame["method_label"]).unique())
    positions, labels, lookup = _bucket_positions(methods)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(methods) * 4), 10))
    for axis, metric, title in zip(
        axes,
        PRIMARY_METRICS,
        ("Lexical diversity (1 - Self-BLEU)", "Semantic diversity (mean 1 - cosine)"),
    ):
        box_values: list[np.ndarray] = []
        used_positions: list[float] = []
        for method in methods:
            for bucket in BUCKET_ORDER:
                mask = (cast(pd.Series, frame["method_label"]) == method) & (
                    cast(pd.Series, frame["bucket"]) == bucket
                )
                subset = cast(pd.Series, frame.loc[mask, metric]).dropna()
                if len(subset):
                    box_values.append(subset.to_numpy(float))
                    used_positions.append(lookup[(method, bucket)])
        axis.boxplot(box_values, positions=used_positions, widths=0.7, showfliers=False)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    _save_figure(fig, output_dir, "diversity_boxplots")

    fig, axes = plt.subplots(len(methods), 2, figsize=(11, max(3.0, 2.8 * len(methods))), squeeze=False)
    colors = dict(zip(BUCKET_ORDER, ("#4c78a8", "#f58518", "#e45756")))
    for row_idx, method in enumerate(methods):
        for col_idx, metric in enumerate(PRIMARY_METRICS):
            axis = axes[row_idx, col_idx]
            for bucket in BUCKET_ORDER:
                mask = (cast(pd.Series, frame["method_label"]) == method) & (
                    cast(pd.Series, frame["bucket"]) == bucket
                )
                ecdf_values = cast(pd.Series, frame.loc[mask, metric]).dropna().sort_values().to_numpy(float)
                if len(ecdf_values):
                    axis.step(
                        ecdf_values,
                        np.arange(1, len(ecdf_values) + 1) / len(ecdf_values),
                        where="post",
                        label=bucket,
                        color=colors[bucket],
                    )
            axis.set_title(f"{method}: {metric.replace('_', ' ')}")
            axis.set_ylabel("ECDF")
            axis.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 1:
                axis.legend(fontsize=8)
    _save_figure(fig, output_dir, "diversity_ecdfs")


def plot_gain(frame: pd.DataFrame, output_dir: Path, analysis_seed: int) -> None:
    methods = sorted(cast(pd.Series, frame["method_label"]).unique())
    rng = np.random.default_rng(analysis_seed)
    fig, axes = plt.subplots(len(methods), 2, figsize=(11, max(3.0, 2.8 * len(methods))), squeeze=False)
    for row_idx, method in enumerate(methods):
        method_frame = cast(
            pd.DataFrame,
            frame.loc[cast(pd.Series, frame["method_label"]) == method],
        )
        for col_idx, metric in enumerate(PRIMARY_METRICS):
            axis = axes[row_idx, col_idx]
            jitter = rng.normal(0.0, 0.025, size=len(method_frame))
            axis.scatter(
                method_frame[metric],
                method_frame["marginal_gain"] + jitter,
                s=7,
                alpha=0.18,
                rasterized=True,
            )
            axis.set_yticks((-1, 0, 1))
            axis.set_ylim(-1.15, 1.15)
            axis.set_xlabel(metric.replace("_", " "))
            axis.set_ylabel("pass@k - baseline pass@1")
            axis.set_title(method)
            axis.grid(alpha=0.2)
    _save_figure(fig, output_dir, "diversity_vs_accuracy_delta")


def _binned_recovery(
    frame: pd.DataFrame,
    metric: str,
    bins: int = 8,
    bootstrap_reps: int = 500,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    valid = frame[["prompt_id", metric, "observed_passk"]].dropna().copy()
    metric_series = cast(pd.Series, valid[metric])
    if metric_series.nunique() < 2:
        return pd.DataFrame()
    valid["bin"] = pd.qcut(metric_series, q=min(bins, metric_series.nunique()), duplicates="drop")
    generator = rng or np.random.default_rng(20260727)
    rows: list[dict[str, float | int]] = []
    for _, bin_frame in valid.groupby("bin", observed=True):
        bin_frame = cast(pd.DataFrame, bin_frame)
        clusters = [
            group
            for _, group in bin_frame.groupby("prompt_id", sort=True)
        ]
        bootstrap = np.empty(bootstrap_reps, dtype=float)
        for rep in range(bootstrap_reps):
            sampled = generator.integers(0, len(clusters), size=len(clusters))
            bootstrap[rep] = float(
                np.mean(
                    np.concatenate(
                        [
                            cast(pd.Series, clusters[idx]["observed_passk"]).to_numpy(float)
                            for idx in sampled
                        ],
                    ),
                ),
            )
        rows.append(
            {
                "diversity": float(cast(pd.Series, bin_frame[metric]).mean()),
                "recovery": float(cast(pd.Series, bin_frame["observed_passk"]).mean()),
                "n": len(bin_frame),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
            },
        )
    return pd.DataFrame(rows)


def plot_recovery_curves(frame: pd.DataFrame, output_dir: Path) -> None:
    hard = cast(pd.DataFrame, frame.loc[cast(pd.Series, frame["baseline_pass1"]) == 0])
    methods = sorted(cast(pd.Series, hard["method_label"]).unique())
    rng = np.random.default_rng(20260727)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, PRIMARY_METRICS):
        for method in methods:
            method_frame = cast(
                pd.DataFrame,
                hard.loc[cast(pd.Series, hard["method_label"]) == method],
            )
            curve = _binned_recovery(method_frame, metric, rng=rng)
            if curve.empty:
                continue
            axis.plot(curve["diversity"], curve["recovery"], marker="o", label=method)
            axis.fill_between(
                curve["diversity"],
                curve["ci_low"],
                curve["ci_high"],
                alpha=0.12,
            )
        axis.set_xlabel(metric.replace("_", " "))
        axis.set_ylabel("Observed recovery probability")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    _save_figure(fig, output_dir, "hard_recovery_curves")


def plot_cross_method_matrix(matrix: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        "recovery_rate",
        "lexical_diversity",
        "semantic_diversity",
        "unique_fraction",
        "recovered_lexical_diversity",
        "recovered_semantic_diversity",
    ]
    methods = matrix["method_label"].tolist()
    values = matrix.set_index("method_label")[metrics].T.to_numpy(float)
    row_min = np.nanmin(values, axis=1, keepdims=True)
    row_span = np.nanmax(values, axis=1, keepdims=True) - row_min
    normalized = (values - row_min) / np.where(row_span == 0, 1, row_span)
    fig, axis = plt.subplots(figsize=(max(7.0, len(methods) * 1.8), 5.5))
    image = axis.imshow(normalized, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=9)
    axis.axhline(3.5, color="black", linewidth=1.5)
    axis.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    axis.set_yticks(range(len(metrics)), [metric.replace("_", " ") for metric in metrics])
    axis.set_title("Fixed-hard metrics (top) and recovered-only descriptive metrics (bottom)")
    fig.colorbar(image, ax=axis, label="Row-normalized value")
    _save_figure(fig, output_dir, "cross_method_matrix")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join((header, separator, *rows))


def write_report(  # noqa: PLR0913
    output_dir: Path,
    validation: dict[str, Any],
    matrix: pd.DataFrame,
    tests: pd.DataFrame,
    model_id: str,
    fingerprint: str,
) -> None:
    primary_mask = (cast(pd.Series, tests["analysis_type"]) == "recovered_minus_unsolved") & cast(
        pd.Series,
        tests["metric"],
    ).isin(PRIMARY_METRICS)
    primary = cast(
        pd.DataFrame,
        tests.loc[
            primary_mask,
            ["method_a", "metric", "n", "effect", "ci_low", "ci_high", "p_adjusted_bh"],
        ],
    )
    matrix_columns = [
        "method_label",
        "hard_n",
        "recovery_rate",
        "lexical_diversity",
        "semantic_diversity",
        "recovered_n",
        "recovered_lexical_diversity",
        "recovered_semantic_diversity",
    ]
    lines = [
        "# GSM8K diversity-recovery analysis",
        "",
        "## Inputs",
        "",
        f"- Files: {validation['files']}",
        f"- Comparison families: {validation['families']}",
        f"- Family-seed replicates: {validation['replicates']}",
        f"- Methods: {', '.join(validation['methods'])}",
        f"- Prompt counts: {validation['prompt_counts']}",
        f"- Raw candidate counts: {validation['candidate_counts']}",
        f"- Compared candidates per prompt: {validation['selected_candidate_count']}",
        f"- Grouped methods: {', '.join(validation['grouped_methods'])}",
        "- Candidate selection: one seeded-random representative per contiguous grouped-method lineage; "
        "the same number sampled without replacement from independent methods. Baseline candidate 0 "
        "is reserved exclusively for the pass@1 hardness anchor and excluded from recovery/diversity.",
        f"- Embedding model: `{model_id}`",
        f"- Embedding fingerprint: `{fingerprint}`",
        "",
        "## Fixed-hard and recovered-only matrix",
        "",
        "The last three columns condition on recovery and are descriptive; "
        "they are not fair standalone method comparisons.",
        "",
        _markdown_table(matrix, matrix_columns),
        "",
        "## Recovered-minus-unsolved tests",
        "",
        "Effects are mean diversity in recovered hard prompts minus mean diversity in unsolved hard prompts.",
        "",
        _markdown_table(primary, list(primary.columns)),
        "",
        "## Interpretation guardrails",
        "",
        "- `pass@1` is candidate 0 from the matched independent-sampling baseline, "
        "not the aggregate unbiased estimator.",
        "- `pass@k` is the observed binary indicator that any evaluated candidate is correct; "
        "for the baseline, these are additional candidates sampled from indices 1 onward.",
        "- Positive associations support a relationship between diversity and recovery; "
        "they do not establish that diversity directly causes accuracy gains.",
        "- Null and opposite-direction effects are reported without alteration.",
        "- Recovered-only method comparisons are selection-conditioned and therefore descriptive.",
        "",
        "Machine-readable prompt metrics, summaries, tests, and figure source values are included beside this report.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    runs = discover_runs(args.results_root, baseline_method=args.baseline_method)
    validation = validation_summary(runs)
    selection = candidate_selection_layout(runs, args.baseline_method)
    validation["selected_candidate_count"] = selection.target_k
    validation["grouped_methods"] = list(selection.grouped_methods)
    validation["candidate_selection"] = "seeded_random_with_held_out_pass1_anchor"
    validation["selection_seed"] = args.analysis_seed
    defaults = load_analysis_defaults(args.config)
    cos_model_id = args.cos_model_id or defaults.cos_model_id
    model_cache_dir = args.model_cache_dir or defaults.model_cache_dir
    model_cache_dir = model_cache_dir.expanduser()
    if not model_cache_dir.is_absolute():
        model_cache_dir = (Path.cwd() / model_cache_dir).resolve()
    validation["config_path"] = str(args.config.expanduser().resolve())
    validation["cos_model_id"] = cos_model_id
    validation["model_cache_dir"] = str(model_cache_dir)
    print(json.dumps(validation, indent=2))
    if args.validate_only:
        return validation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_cache_dir = args.analysis_cache_dir or args.output_dir / "cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(analysis_cache_dir / "analysis_cache.sqlite")
    try:
        frame, fingerprint = build_prompt_rows(
            runs,
            cache,
            cos_model_id,
            args.device,
            args.batch_size,
            args.num_workers,
            model_cache_dir,
            args.analysis_seed,
            baseline_method=args.baseline_method,
        )
    finally:
        cache.close()

    summary, matrix = build_summary(frame)
    tests = run_inference(frame, args.bootstrap_reps, args.permutation_reps, args.analysis_seed)
    write_csv(frame, args.output_dir / "prompt_metrics.csv")
    write_csv(summary, args.output_dir / "summary.csv")
    write_csv(matrix, args.output_dir / "cross_method_matrix.csv")
    write_csv(tests, args.output_dir / "statistical_tests.csv")
    tests_json = tests.to_json(orient="records")
    if tests_json is None:
        raise ValueError("failed to serialize statistical tests")
    test_records = json.loads(tests_json)
    (args.output_dir / "statistical_tests.json").write_text(json.dumps(test_records, indent=2))
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    plot_bucket_distributions(frame, args.output_dir)
    plot_gain(frame, args.output_dir, args.analysis_seed)
    plot_recovery_curves(frame, args.output_dir)
    plot_cross_method_matrix(matrix, args.output_dir)
    write_report(args.output_dir, validation, matrix, tests, cos_model_id, fingerprint)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cos-model-id", default=None, help="Override config.cos_model_id")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Override config.cache_dir for model loading",
    )
    parser.add_argument("--analysis-cache-dir", type=Path, default=None, help="SQLite metric/embedding cache")
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=20260727)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("batch_size", "num_workers", "bootstrap_reps", "permutation_reps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    run_analysis(args)


if __name__ == "__main__":
    main()

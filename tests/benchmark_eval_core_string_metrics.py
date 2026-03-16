#!/usr/bin/env python3
"""
CPU benchmark for eval_core string metrics.

Uses a math-results JSON file and compares the current implementation in
`d5p4.eval_core` against a preserved baseline of the original algorithms.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping

import sacrebleu

from d5p4 import eval_core


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


def _baseline_compute_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _baseline_compute_distinct_metrics(
    texts: list[str],
    vocab_size: int | None = None,
    references_for_vocab: list[str] | None = None,
) -> dict[str, float]:
    def ngrams_with_left_pad(tokens: list[str], size: int) -> list[tuple[str, ...]]:
        padded = ["<s>"] * (size - 1) + tokens
        return [tuple(padded[i : i + size]) for i in range(len(tokens))]

    if not texts:
        return {}

    tokenizer = _FakeTokenizer()
    if vocab_size is None and references_for_vocab is not None:
        vocab = set()
        for sentence in references_for_vocab:
            vocab.update(tokenizer.tokenize(sentence))
        vocab_size = len(vocab)

    distinct_tokens = set()
    distinct_tokens_2grams: set[tuple[str, ...]] = set()
    distinct_tokens_3grams: set[tuple[str, ...]] = set()
    total_tokens: list[str] = []
    total_tokens_2grams: list[tuple[str, ...]] = []
    total_tokens_3grams: list[tuple[str, ...]] = []

    for prediction in texts:
        tokens = tokenizer.tokenize(prediction)
        tokens_2grams = ngrams_with_left_pad(tokens, 2)
        tokens_3grams = ngrams_with_left_pad(tokens, 3)

        distinct_tokens.update(tokens)
        distinct_tokens_2grams.update(tokens_2grams)
        distinct_tokens_3grams.update(tokens_3grams)

        total_tokens.extend(tokens)
        total_tokens_2grams.extend(tokens_2grams)
        total_tokens_3grams.extend(tokens_3grams)

    metrics = {}
    metrics["distinct_1"] = len(distinct_tokens) / len(total_tokens) if total_tokens else 0.0
    metrics["distinct_2"] = len(distinct_tokens_2grams) / len(total_tokens_2grams) if total_tokens_2grams else 0.0
    metrics["distinct_3"] = len(distinct_tokens_3grams) / len(total_tokens_3grams) if total_tokens_3grams else 0.0

    if vocab_size is not None and total_tokens:
        try:
            ead = len(distinct_tokens) / (vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** len(total_tokens)))
            metrics["expectation_adjusted_distinct"] = ead
        except ZeroDivisionError:
            metrics["expectation_adjusted_distinct"] = 0.0

    return metrics


def _baseline_compute_self_bleu(texts: list[str]) -> float:
    if len(texts) <= 1:
        return 0.0

    bleu_scores = []
    for i, hypothesis in enumerate(texts):
        references = [texts[j] for j in range(len(texts)) if j != i]
        if not references:
            continue

        bleu = sacrebleu.sentence_bleu(hypothesis, references)
        bleu_scores.append(bleu.score)

    if not bleu_scores:
        return 0.0

    return sum(bleu_scores) / len(bleu_scores)


def _baseline_group_reference_alignment(args: tuple[list[str], list[str]]) -> tuple[list[float], float, float]:
    preds, refs = args
    f1_scores = [max([_baseline_compute_f1(pred, ref) for ref in refs]) if refs else 0.0 for pred in preds]

    best_f1_for_question = max(f1_scores) if f1_scores else 0.0

    best_bleu_for_question = 0.0
    for pred in preds:
        bleu_result = sacrebleu.sentence_bleu(pred, refs)
        best_bleu_for_question = max(best_bleu_for_question, bleu_result.score)

    return f1_scores, best_f1_for_question, best_bleu_for_question


class _BaselineStringMetrics:
    def compute_distinct_metrics(
        self,
        texts: list[str],
        vocab_size: int | None = None,
        references_for_vocab: list[str] | None = None,
    ) -> dict[str, float]:
        return _baseline_compute_distinct_metrics(
            texts,
            vocab_size=vocab_size,
            references_for_vocab=references_for_vocab,
        )

    def compute_self_bleu(self, texts: list[str]) -> float:
        return _baseline_compute_self_bleu(texts)

    def _vocab_size_from_references(self, references_for_vocab: list[str] | None) -> int | None:
        if references_for_vocab is None:
            return None
        vocab = set()
        tokenizer = _FakeTokenizer()
        for sentence in references_for_vocab:
            vocab.update(tokenizer.tokenize(sentence))
        return len(vocab)

    def diversity_grouped(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
    ) -> dict[str, float]:
        all_metrics = {}

        vocab_ref_tokens = []
        if references and any(refs for refs in references):
            for sublist in references:
                vocab_ref_tokens.extend(sublist)

        valid_groups = [group for group in predictions if group]
        references_for_vocab = vocab_ref_tokens if vocab_ref_tokens else None
        distinct_metrics_list = []
        self_bleu_scores = []
        for group in valid_groups:
            d_metrics = self.compute_distinct_metrics(group, references_for_vocab=references_for_vocab)
            distinct_metrics_list.append(d_metrics)
            self_bleu_scores.append(self.compute_self_bleu(group))

        if distinct_metrics_list:
            keys = distinct_metrics_list[0].keys()
            for key in keys:
                values = [m[key] for m in distinct_metrics_list if key in m]
                all_metrics.update(eval_core.compute_statistics(values, key))
        all_metrics.update(eval_core.compute_statistics(self_bleu_scores, "self_bleu"))
        return all_metrics

    def reference_alignment(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
    ) -> dict[str, float | str]:
        all_metrics = {}
        if not (references and any(refs for refs in references)):
            return all_metrics

        grouped_pairs = [(preds, refs) for preds, refs in zip(predictions, references) if preds and refs]
        flattened_predictions = [pred for preds, _ in grouped_pairs for pred in preds]
        flattened_references = [refs for preds, refs in grouped_pairs for _ in preds]

        f1_scores = []
        f1_at_k_scores = []
        bleu_at_k_scores = []
        for pair in grouped_pairs:
            group_f1_scores, best_f1_for_question, best_bleu_for_question = _baseline_group_reference_alignment(pair)
            f1_scores.extend(group_f1_scores)
            f1_at_k_scores.append(best_f1_for_question)
            bleu_at_k_scores.append(best_bleu_for_question)

        all_metrics.update(eval_core.compute_statistics(f1_scores, "f1"))

        bleu_score = 0.0
        if flattened_references:
            max_refs = max(len(refs) for refs in flattened_references)
            formatted_refs = []
            for i in range(max_refs):
                ref_list = []
                for refs in flattened_references:
                    ref_list.append(refs[i] if i < len(refs) else refs[0])
                formatted_refs.append(ref_list)
            bleu_score = sacrebleu.corpus_bleu(flattened_predictions, formatted_refs).score

        all_metrics["bleu"] = bleu_score
        k = len(predictions[0]) if predictions and predictions[0] else 0
        all_metrics["k"] = float(k)
        if k > 0:
            all_metrics.update(eval_core.compute_statistics(f1_at_k_scores, "f1_at_k"))
            all_metrics.update(eval_core.compute_statistics(bleu_at_k_scores, "bleu_at_k"))
        return all_metrics

    def forward(self, predictions: list[list[str]], references: list[list[str]]) -> dict[str, float | str]:
        return {
            **self.reference_alignment(predictions, references),
            **self.diversity_grouped(predictions, references),
        }


def _load_math_results(path: Path, repeat_groups: int) -> tuple[list[list[str]], list[list[str]]]:
    payload = json.loads(path.read_text())
    results = payload["results"]
    predictions = [row["generations"] for row in results] * repeat_groups
    references = [[row["gold_answer"]] for row in results] * repeat_groups
    return predictions, references


def _benchmark(name: str, fn, warmup: int, repeat: int) -> dict[str, float | str]:
    for _ in range(warmup):
        fn()

    timings_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        timings_ms.append(float((time.perf_counter() - start) * 1000))

    return {
        "name": name,
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def _print_result(result: Mapping[str, float | str]) -> None:
    print(
        f"{result['name']:<28} mean={float(result['mean_ms']):9.2f} ms  "
        f"median={float(result['median_ms']):9.2f} ms  "
        f"min={float(result['min_ms']):9.2f} ms  max={float(result['max_ms']):9.2f} ms",
    )


def _assert_close_enough(baseline: Mapping[str, float | str], optimized: Mapping[str, float | str]) -> None:
    for key in ("distinct_1", "distinct_2", "distinct_3", "self_bleu", "f1", "f1_at_k", "bleu", "bleu_at_k"):
        if key not in baseline or key not in optimized:
            continue
        if abs(float(baseline[key]) - float(optimized[key])) > 1e-9:
            raise AssertionError(f"metric mismatch for {key}: baseline={baseline[key]} optimized={optimized[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--repeat", type=int, default=7, help="Benchmark repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument("--repeat-groups", type=int, default=4, help="Duplicate the dataset to amplify CPU work.")
    args = parser.parse_args()

    predictions, references = _load_math_results(args.input, args.repeat_groups)

    eval_core._STRING_METRICS_TOKENIZER = _FakeTokenizer()  # type: ignore
    eval_core._cached_metric_tokenize.cache_clear()
    eval_core._cached_lower_split_tokens.cache_clear()
    eval_core._cached_lower_split_counter.cache_clear()

    baseline = _BaselineStringMetrics()
    optimized = eval_core.StringMetrics()

    baseline_forward = baseline.forward(predictions, references)
    optimized_forward = optimized.forward(predictions, references)
    _assert_close_enough(baseline_forward, optimized_forward)

    benchmarks = [
        _benchmark(
            "baseline diversity_grouped",
            lambda: baseline.diversity_grouped(predictions, references),
            args.warmup,
            args.repeat,
        ),
        _benchmark(
            "optimized diversity_grouped",
            lambda: optimized.diversity_grouped(predictions, references),
            args.warmup,
            args.repeat,
        ),
        _benchmark(
            "baseline reference_align",
            lambda: baseline.reference_alignment(predictions, references),
            args.warmup,
            args.repeat,
        ),
        _benchmark(
            "optimized reference_align",
            lambda: optimized.reference_alignment(predictions, references),
            args.warmup,
            args.repeat,
        ),
        _benchmark("baseline forward", lambda: baseline.forward(predictions, references), args.warmup, args.repeat),
        _benchmark("optimized forward", lambda: optimized.forward(predictions, references), args.warmup, args.repeat),
    ]

    print(f"dataset groups={len(predictions)}  generations/group={len(predictions[0]) if predictions else 0}")
    for result in benchmarks:
        _print_result(result)

    for slow_name, fast_name in (
        ("baseline diversity_grouped", "optimized diversity_grouped"),
        ("baseline reference_align", "optimized reference_align"),
        ("baseline forward", "optimized forward"),
    ):
        slow = next(result for result in benchmarks if result["name"] == slow_name)
        fast = next(result for result in benchmarks if result["name"] == fast_name)
        speedup = float(slow["mean_ms"]) / float(fast["mean_ms"]) if float(fast["mean_ms"]) else float("inf")
        print(f"speedup {fast_name:<28} x{speedup:.2f}")


if __name__ == "__main__":
    main()

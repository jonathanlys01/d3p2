#!/usr/bin/env python
"""Read Dream GSM8K result JSONs (E2 or B5) and report accuracy under both extractors and answer lengths.

CPU only, no model load: GSM8K scoring is string matching. For each run directory under the given roots:
per-answer accuracy with lm-eval flexible-extract (Dream's published protocol), strict-match and the universal parser,
pass@k of the k first samples with flexible-extract, the share of bare-number answers, and the median answer length
(tokens before the first end token when the run recorded `answer_tokens`, else characters).

Usage (login node):
    python .scripts_next/dream_check_report.py $WORK_ejh/projects/d3p2_waves/b5/results \\
        $WORK_ejh/projects/d3p2_waves/e2/results/dream_gsm8k --first 50
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from d5p4.eval_core import MathEvaluator
from d5p4.text_postprocessors import lm_eval_gsm8k_extract, lm_eval_gsm8k_normalize


BARE_NUMBER = re.compile(r"\s*\$?-?[\d,.]+\s*\.?\s*")


def result_file(run_dir: Path) -> Path | None:
    files = [path for path in sorted(run_dir.glob("math-*.json")) if "-math-bon-" not in path.name]
    return files[-1] if files else None


def report(path: Path, first: int | None) -> dict[str, object]:
    results = json.loads(path.read_text())["results"]
    if first is not None:
        results = results[:first]
    flexible = MathEvaluator(answer_extractor="lm_eval_flexible")
    universal = MathEvaluator()
    answers = [(row, text) for row in results for text in row["generations"]]
    n = len(answers)
    k = min(len(row["generations"]) for row in results)
    flex_scores = [flexible.score_group(row["generations"], row["gold_answer"]) for row in results]
    tokens = [count for row in results for count in row.get("answer_tokens", [])]
    return {
        "questions": len(results),
        "answers": n,
        "acc_flexible": sum(map(sum, flex_scores)) / n,
        "acc_strict": sum(
            lm_eval_gsm8k_normalize(lm_eval_gsm8k_extract(text, mode="strict"))
            == lm_eval_gsm8k_normalize(str(row["gold_answer"]))
            for row, text in answers
        )
        / n,
        "acc_universal": sum(universal.check(text, row["gold_answer"]) for row, text in answers) / n,
        f"pass@{k}_flexible": sum(any(scores) for scores in flex_scores) / len(results),
        "bare_number": sum(bool(BARE_NUMBER.fullmatch(text)) for _, text in answers) / n,
        "median_answer": f"{statistics.median(tokens)} tok"
        if tokens
        else f"{statistics.median(len(text) for _, text in answers)} chars",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", type=Path, nargs="+", help="results directories (searched for run directories)")
    parser.add_argument("--first", type=int, default=None, help="only the first N questions of each run")
    args = parser.parse_args()
    for root in args.roots:
        for run_dir in sorted({path.parent for path in root.rglob("math-*.json")}):
            if (path := result_file(run_dir)) is None:
                continue
            row = report(path, args.first)
            cells = " ".join(
                f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()
            )
            print(f"{run_dir.relative_to(root.parent)}: {cells}")


if __name__ == "__main__":
    main()

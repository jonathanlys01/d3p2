"""Python code extraction, parsing, and benchmark-test evaluation."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from d5p4.eval_utils import compute_statistics


_FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CodeValidationResult:
    extracted_code: str
    full_code: str
    parse_ok: bool
    passed: bool
    status: str
    error: str = ""
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_python_code(text: str) -> str:
    """Extract Python code from a model generation.

    Prefer fenced Python blocks, then any fenced block, then the raw generation.
    """
    if not text:
        return ""

    matches = list(_FENCED_CODE_RE.finditer(text))
    if matches:
        python_match = next(
            (match for match in matches if match.group("lang").lower() in {"python", "py"}),
            None,
        )
        match = python_match if python_match is not None else matches[0]
        return match.group("code").strip("\r\n")

    return text.strip("\r\n")


def validate_python_ast(code: str) -> tuple[bool, str]:
    """Return whether *code* parses as Python, plus an error message."""
    try:
        ast.parse(code)
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"
    except ValueError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, ""


def _format_tests(tests: list[str], entry_point: str) -> str:
    test_code = "\n".join(test for test in tests if test)
    if entry_point:
        return f"{test_code}\n\ncheck({entry_point})\n"
    return f"{test_code}\n"


def _build_execution_script(full_code: str, tests: list[str], entry_point: str) -> str:
    return f"import faulthandler\nfaulthandler.enable()\n\n{full_code}\n\n{_format_tests(tests, entry_point)}"


def _run_script(script: str, timeout_s: float) -> tuple[str, str, str]:
    with tempfile.TemporaryDirectory(prefix="d5p4-code-eval-") as tmpdir:
        script_path = Path(tmpdir) / "candidate_eval.py"
        script_path.write_text(script, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return "timeout", stdout, stderr

    if completed.returncode == 0:
        return "passed", completed.stdout, completed.stderr
    return "failed", completed.stdout, completed.stderr


class CodeEvaluator:
    """Evaluate generated Python code against HumanEval/MBPP-style tests."""

    def __init__(self, timeout_s: float = 5.0):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s

    @staticmethod
    def build_full_code(generation: str, prompt: str, dataset: str) -> tuple[str, str]:
        extracted = extract_python_code(generation)
        if dataset == "humaneval":
            return extracted, f"{prompt.rstrip()}\n{extracted}\n"
        if dataset == "mbpp":
            return extracted, f"{extracted}\n"
        raise ValueError(f"Unsupported code dataset {dataset!r}")

    def validate(
        self,
        generation: str,
        *,
        prompt: str,
        tests: list[str],
        entry_point: str,
        dataset: str,
    ) -> CodeValidationResult:
        extracted, full_code = self.build_full_code(generation, prompt, dataset)
        parse_ok, parse_error = validate_python_ast(full_code)
        if not parse_ok:
            return CodeValidationResult(
                extracted_code=extracted,
                full_code=full_code,
                parse_ok=False,
                passed=False,
                status="parse_error",
                error=parse_error,
            )

        script = _build_execution_script(full_code, tests, entry_point)
        status, stdout, stderr = _run_script(script, self.timeout_s)
        passed = status == "passed"
        error = "" if passed else (stderr.strip() or stdout.strip() or status)
        return CodeValidationResult(
            extracted_code=extracted,
            full_code=full_code,
            parse_ok=True,
            passed=passed,
            status=status,
            error=error,
            stdout=stdout,
            stderr=stderr,
        )

    def score_group(
        self,
        generations: list[str],
        *,
        prompt: str,
        tests: list[str],
        entry_point: str,
        dataset: str,
    ) -> list[CodeValidationResult]:
        return [
            self.validate(
                generation,
                prompt=prompt,
                tests=tests,
                entry_point=entry_point,
                dataset=dataset,
            )
            for generation in generations
        ]

    @staticmethod
    def accuracy(results: list[CodeValidationResult]) -> float:
        return sum(result.passed for result in results) / len(results) if results else 0.0

    @staticmethod
    def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
        if n < k:
            return float("nan")
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        num, den = 1.0, 1.0
        for i in range(k):
            num *= n - c - i
            den *= n - i
        return 1.0 - num / den

    def evaluate(
        self,
        validation_groups: list[list[CodeValidationResult]],
        k_values: list[int] | None = None,
    ) -> dict[str, float | str]:
        if not validation_groups:
            return {}

        group_size = max(len(group) for group in validation_groups)
        if k_values is None:
            k_values = [1, 2, 3, 4, 8, 16]

        effective_ks: list[int] = []
        seen: set[int] = set()
        for k in k_values:
            if 1 <= k <= group_size and k not in seen:
                effective_ks.append(k)
                seen.add(k)

        accuracies = [self.accuracy(group) for group in validation_groups if group]
        flat = [result for group in validation_groups for result in group]
        parse_values = [float(result.parse_ok) for result in flat]
        test_values = [float(result.passed) for result in flat]

        metrics: dict[str, float | str] = {}
        metrics.update(compute_statistics(accuracies, "accuracy"))
        metrics.update(compute_statistics(parse_values, "parse_success_rate"))
        metrics.update(compute_statistics(test_values, "test_success_rate"))
        metrics["k"] = float(group_size)

        for k in effective_ks:
            vals = []
            for group in validation_groups:
                n = len(group)
                c = sum(result.passed for result in group)
                val = self._pass_at_k_estimator(n, c, k)
                if not np.isnan(val):
                    vals.append(val)
            stats = compute_statistics(vals, f"pass_at_{k}")
            metrics.update(stats)
            metrics[f"pass@{k}"] = stats[f"pass_at_{k}"]

        summary_parts = [
            f"Acc: {metrics.get('accuracy', float('nan')):.4f}",
            f"Parse: {metrics.get('parse_success_rate', float('nan')):.4f}",
            f"Tests: {metrics.get('test_success_rate', float('nan')):.4f}",
        ]
        for k in effective_ks:
            val = metrics.get(f"pass_at_{k}", float("nan"))
            if isinstance(val, float) and not np.isnan(val):
                summary_parts.append(f"pass@{k}: {val:.4f}")
        metrics["code_metrics_summary"] = " | ".join(summary_parts)
        return metrics


def validation_results_to_json(results: list[CodeValidationResult]) -> list[dict[str, Any]]:
    """Convert validation results into JSON-safe dictionaries."""
    return [json.loads(json.dumps(result.to_dict())) for result in results]

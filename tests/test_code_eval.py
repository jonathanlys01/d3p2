import os
import sys
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.code_eval import CodeEvaluator, extract_python_code, validate_python_ast


class TestCodeEval(unittest.TestCase):
    def test_extract_python_fenced_block(self):
        text = "Here is the answer:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."

        self.assertEqual(extract_python_code(text), "def add(a, b):\n    return a + b")

    def test_extract_raw_code_fallback(self):
        text = "def add(a, b):\n    return a + b"

        self.assertEqual(extract_python_code(text), text)

    def test_ast_validation_reports_syntax_error(self):
        ok, error = validate_python_ast("def bad(:\n    pass")

        self.assertFalse(ok)
        self.assertIn("SyntaxError", error)

    def test_mbpp_validation_passes_assertions(self):
        evaluator = CodeEvaluator(timeout_s=2.0)

        result = evaluator.validate(
            "def add(a, b):\n    return a + b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
            entry_point="",
            dataset="mbpp",
        )

        self.assertTrue(result.parse_ok)
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "passed")

    def test_mbpp_validation_fails_assertions(self):
        evaluator = CodeEvaluator(timeout_s=2.0)

        result = evaluator.validate(
            "def add(a, b):\n    return a - b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3"],
            entry_point="",
            dataset="mbpp",
        )

        self.assertTrue(result.parse_ok)
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "failed")

    def test_humaneval_validation_uses_prompt_and_entry_point(self):
        evaluator = CodeEvaluator(timeout_s=2.0)
        prompt = "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n"

        result = evaluator.validate(
            "    return a + b",
            prompt=prompt,
            tests=["def check(candidate):\n    assert candidate(2, 3) == 5"],
            entry_point="add",
            dataset="humaneval",
        )

        self.assertTrue(result.parse_ok)
        self.assertTrue(result.passed)

    def test_validation_reports_timeout(self):
        evaluator = CodeEvaluator(timeout_s=0.2)

        result = evaluator.validate(
            "def loop():\n    while True:\n        pass",
            prompt="Write a looping function.",
            tests=["loop()"],
            entry_point="",
            dataset="mbpp",
        )

        self.assertTrue(result.parse_ok)
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "timeout")

    def test_evaluate_reports_pass_at_k_and_rates(self):
        evaluator = CodeEvaluator(timeout_s=2.0)
        passing = evaluator.validate(
            "def add(a, b):\n    return a + b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3"],
            entry_point="",
            dataset="mbpp",
        )
        failing = evaluator.validate(
            "def add(a, b):\n    return a - b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3"],
            entry_point="",
            dataset="mbpp",
        )

        metrics = evaluator.evaluate([[passing, failing]], k_values=[1, 2])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["parse_success_rate"], 1.0)
        self.assertEqual(metrics["test_success_rate"], 0.5)
        self.assertEqual(metrics["pass@2"], 1.0)

    def test_evaluate_default_k_values_include_three(self):
        evaluator = CodeEvaluator(timeout_s=2.0)
        passing = evaluator.validate(
            "def add(a, b):\n    return a + b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3"],
            entry_point="",
            dataset="mbpp",
        )
        failing = evaluator.validate(
            "def add(a, b):\n    return a - b",
            prompt="Write a function to add two numbers.",
            tests=["assert add(1, 2) == 3"],
            entry_point="",
            dataset="mbpp",
        )

        metrics = evaluator.evaluate([[passing, failing, failing]])

        self.assertIn("pass@3", metrics)
        self.assertNotIn("pass@4", metrics)


if __name__ == "__main__":
    unittest.main()

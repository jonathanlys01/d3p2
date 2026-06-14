import os
import sys
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.result_schema import (
    INTERNAL_SCORES,
    LEGACY_INTERNAL_SCORES,
    build_generation_result_payload,
    get_eval_text_groups,
    payload_tree_lines,
    validate_generation_result_payload,
)


class TestResultSchema(unittest.TestCase):
    def test_build_payload_uses_canonical_internal_scores_only(self):
        payload = build_generation_result_payload(
            text_samples=[["a0", "a1"], ["b0", "b1"]],
            eval_text_samples=[["a1"], ["b0"]],
            config={"model": "llada"},
            internal_scores=[[0.1, 0.8], [0.7, 0.2]],
            eval_selected_indices=[[1], [0]],
            internal_score_metadata={"name": "confidence"},
            experiment_id="abc",
        )

        self.assertEqual(payload[INTERNAL_SCORES], [[0.1, 0.8], [0.7, 0.2]])
        self.assertNotIn(LEGACY_INTERNAL_SCORES, payload)
        self.assertEqual(get_eval_text_groups(payload), [["a1"], ["b0"]])

    def test_legacy_internal_scores_are_normalized(self):
        payload = {
            "text_samples": [["a0", "a1"]],
            "config": {"model": "llada"},
            LEGACY_INTERNAL_SCORES: [[0.1, 0.2]],
        }

        validate_generation_result_payload(payload)

        self.assertEqual(payload[INTERNAL_SCORES], [[0.1, 0.2]])

    def test_canonical_internal_scores_are_not_duplicated_during_normalization(self):
        payload = {
            "text_samples": [["a0", "a1"]],
            "config": {"model": "llada"},
            INTERNAL_SCORES: [[0.1, 0.2]],
        }

        validate_generation_result_payload(payload)

        self.assertNotIn(LEGACY_INTERNAL_SCORES, payload)

    def test_internal_scores_must_align_with_text_samples(self):
        payload = {
            "text_samples": [["a0", "a1"]],
            "config": {"model": "llada"},
            INTERNAL_SCORES: [[0.1]],
        }

        with self.assertRaisesRegex(ValueError, "one score per sequence"):
            validate_generation_result_payload(payload)

    def test_internal_scores_must_be_finite(self):
        payload = {
            "text_samples": [["a0"]],
            "config": {"model": "llada"},
            INTERNAL_SCORES: [[float("inf")]],
        }

        with self.assertRaises(ValueError):
            validate_generation_result_payload(payload)

    def test_payload_tree_lines_prints_payload_structure(self):
        payload = build_generation_result_payload(
            text_samples=[["a0", "a1"]],
            config={"model": "llada"},
            internal_scores=[[0.1, 0.2]],
        )

        tree = "\n".join(payload_tree_lines(payload, name="sample.json", max_items=3))

        self.assertIn("sample.json: dict", tree)
        self.assertIn("text_samples: list[1]", tree)
        self.assertIn("config: dict", tree)
        self.assertIn("internal_scores: list[1]", tree)

    def test_code_eval_payload_uses_existing_schema_fields(self):
        payload = build_generation_result_payload(
            text_samples=[["def add(a, b):\n    return a + b"]],
            config={"model": "llada", "code_dataset": "mbpp"},
            references=[["def add(a, b):\n    return a + b"]],
            internal_scores=[[0.9]],
            internal_score_metadata={"name": "confidence"},
            metrics={"accuracy": 1.0, "code_metrics_summary": "Acc: 1.0000"},
            experiment_id="code-run",
            extra={
                "results": [
                    {
                        "task_id": "1",
                        "prompt": "Write a function to add two numbers.",
                        "reference_code": "def add(a, b):\n    return a + b",
                        "tests": ["assert add(1, 2) == 3"],
                        "entry_point": "",
                        "dataset": "mbpp",
                        "generations": ["def add(a, b):\n    return a + b"],
                        "validation": [{"parse_ok": True, "passed": True, "status": "passed"}],
                        "scores": [1],
                        "accuracy": 1.0,
                    },
                ],
                "overall_accuracy": 1.0,
                "code_metrics": {"accuracy": 1.0, "code_metrics_summary": "Acc: 1.0000"},
            },
        )

        validate_generation_result_payload(payload)

        self.assertEqual(payload["metrics"]["accuracy"], 1.0)
        self.assertEqual(payload["code_metrics"]["accuracy"], 1.0)
        self.assertEqual(payload["results"][0]["validation"][0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()

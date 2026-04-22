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
    def test_build_payload_uses_internal_scores_with_legacy_alias(self):
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
        self.assertEqual(payload[LEGACY_INTERNAL_SCORES], payload[INTERNAL_SCORES])
        self.assertEqual(get_eval_text_groups(payload), [["a1"], ["b0"]])

    def test_legacy_internal_scores_are_normalized(self):
        payload = {
            "text_samples": [["a0", "a1"]],
            "config": {"model": "llada"},
            LEGACY_INTERNAL_SCORES: [[0.1, 0.2]],
        }

        validate_generation_result_payload(payload)

        self.assertEqual(payload[INTERNAL_SCORES], [[0.1, 0.2]])

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

        tree = "\n".join(payload_tree_lines(payload, name="sample.json", max_items=2))

        self.assertIn("sample.json: dict", tree)
        self.assertIn("text_samples: list[1]", tree)
        self.assertIn("config: dict", tree)
        self.assertIn("internal_scores: list[1]", tree)


if __name__ == "__main__":
    unittest.main()

import json
import os
import sys
import tempfile
import unittest
from pprint import pprint
from unittest.mock import patch


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.eval_core import MathEvaluator, StringMetrics


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class TestEvalCoreStringMetrics(unittest.TestCase):
    def setUp(self):
        tokenizer_patch = patch("d5p4.eval_core.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
        self.addCleanup(tokenizer_patch.stop)
        tokenizer_patch.start()

    def test_text_samples_shape_matches_llada_mdlm_outputs(self):
        payload = {
            "text_samples": [
                [
                    "The capital of France is Paris",
                    "Paris is the capital of France",
                    "France has Paris as its capital",
                ],
                [
                    "2 plus 2 equals 4",
                    "The answer is 4",
                    "It should be four",
                ],
            ],
            "config": {"model": "llada"},
        }
        references = [
            ["Paris"],
            ["4", "four"],
        ]

        metrics = StringMetrics()
        grouped_diversity = metrics.diversity_grouped(payload["text_samples"], references)
        corpus_diversity = metrics.diversity_corpus(payload["text_samples"], references, prefix="batch")
        reference_alignment = metrics.reference_alignment(payload["text_samples"], references)

        print("\nGrouped diversity metrics:")
        pprint(grouped_diversity)
        print("\nCorpus diversity metrics:")
        pprint(corpus_diversity)
        print("\nReference alignment metrics:")
        pprint(reference_alignment)

        self.assertIn("distinct_2", grouped_diversity)
        self.assertIn("self_bleu", grouped_diversity)
        self.assertIn("expectation_adjusted_distinct", grouped_diversity)
        self.assertIn("batch_distinct_2", corpus_diversity)
        self.assertIn("batch_self_bleu", corpus_diversity)
        self.assertIn("batch_expectation_adjusted_distinct", corpus_diversity)
        self.assertIn("f1", reference_alignment)
        self.assertIn("bleu", reference_alignment)
        self.assertIn("f1_at_k", reference_alignment)
        self.assertIn("bleu_at_k", reference_alignment)

        self.assertNotIn("batch_distinct_2", grouped_diversity)
        self.assertGreaterEqual(grouped_diversity["distinct_2"], 0.0)
        self.assertGreaterEqual(corpus_diversity["batch_distinct_2"], 0.0)
        self.assertGreaterEqual(reference_alignment["f1_at_k"], reference_alignment["f1"])

    def test_math_results_shape_persists_string_metrics_without_model_downloads(self):
        payload = {
            "results": [
                {
                    "question": "What is 3 + 4?",
                    "gold_answer": "7",
                    "generations": [
                        "We compute 3 + 4 = 7. Final answer: 7",
                        "The answer is 8",
                        "7",
                    ],
                },
                {
                    "question": "What is 10 / 2?",
                    "gold_answer": "5",
                    "generations": [
                        "10 / 2 = 5",
                        "five",
                        "The answer is 6",
                    ],
                },
            ],
            "config": {"model": "mdlm"},
        }

        evaluator = MathEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "math-results.json")
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=4)

            math_metrics = evaluator.eval_from_file(file_path, force=True, k_values=[1, 2, 3])

            self.assertIsNotNone(math_metrics)
            assert math_metrics is not None
            print("\nMath evaluator metrics:")
            pprint(math_metrics)
            self.assertIn("accuracy", math_metrics)
            self.assertIn("pass@1", math_metrics)
            self.assertIn("pass@2", math_metrics)
            self.assertIn("pass_at_1", math_metrics)
            self.assertIn("pass_at_2", math_metrics)
            self.assertIn("f1", math_metrics)
            self.assertIn("distinct_2", math_metrics)
            self.assertIn("batch_distinct_2", math_metrics)
            self.assertIn("batch_self_bleu", math_metrics)
            self.assertIn("math_metrics_summary", math_metrics)

            with open(file_path) as f:
                saved = json.load(f)

            self.assertIn("math_metrics", saved)
            self.assertEqual(saved["math_metrics"]["batch_distinct_2"], math_metrics["batch_distinct_2"])


if __name__ == "__main__":
    unittest.main()

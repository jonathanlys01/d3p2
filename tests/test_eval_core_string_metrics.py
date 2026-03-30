import json
import os
import sys
import tempfile
import unittest
from pprint import pprint
from unittest.mock import patch

import numpy as np
import torch


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4._oversample_baseline import _select_and_evaluate_baseline
from d5p4.eval_core import Evaluator, MathEvaluator, Perplexity, StringMetrics, _is_math_results_file


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakePerplexityTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = False,
    ) -> _FakeBatch:
        del return_tensors, padding, truncation, add_special_tokens

        encoded = [[1, 2, 3] if text.strip() else [] for text in texts]
        max_len = max((len(ids) for ids in encoded), default=0)
        input_ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            if ids:
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1
        return _FakeBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class _FakePerplexityModel(torch.nn.Module):
    def eval(self):
        return self

    def to(self, _device):
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_dict: bool = True):
        del attention_mask, return_dict
        batch_size, seq_len = input_ids.shape
        vocab_size = 8
        return type(
            "_FakeOutputs",
            (),
            {"last_hidden_state": torch.zeros(batch_size, seq_len, vocab_size)},
        )()


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
        self.assertIn("empirical_entropy", grouped_diversity)
        self.assertIn("self_bleu", grouped_diversity)
        self.assertIn("expectation_adjusted_distinct", grouped_diversity)
        self.assertIn("batch_distinct_2", corpus_diversity)
        self.assertIn("batch_empirical_entropy", corpus_diversity)
        self.assertIn("batch_self_bleu", corpus_diversity)
        self.assertIn("batch_expectation_adjusted_distinct", corpus_diversity)
        self.assertIn("f1", reference_alignment)
        self.assertIn("bleu", reference_alignment)
        self.assertIn("f1_at_k", reference_alignment)
        self.assertIn("bleu_at_k", reference_alignment)

        self.assertNotIn("batch_distinct_2", grouped_diversity)
        self.assertGreaterEqual(grouped_diversity["distinct_2"], 0.0)
        self.assertGreaterEqual(grouped_diversity["empirical_entropy"], 0.0)
        self.assertGreaterEqual(corpus_diversity["batch_distinct_2"], 0.0)
        self.assertGreaterEqual(corpus_diversity["batch_empirical_entropy"], 0.0)
        self.assertGreaterEqual(reference_alignment["f1_at_k"], reference_alignment["f1"])

    def test_empirical_entropy_matches_definition(self):
        metrics = StringMetrics()

        diversity = metrics.diversity_set(["a a b", "b c"], prefix="batch")
        probs = np.array([2 / 5, 2 / 5, 1 / 5], dtype=float)
        expected_entropy = float(-(probs * np.log(probs)).sum())

        self.assertAlmostEqual(diversity["batch_empirical_entropy"], expected_entropy)

    def test_grouped_empirical_entropy_is_sequence_level(self):
        metrics = StringMetrics()

        predictions = [["a a b", "a b c"]]

        grouped = metrics.diversity_grouped(predictions)
        corpus = metrics.diversity_corpus(predictions, prefix="batch")

        seq_probs_1 = np.array([2 / 3, 1 / 3], dtype=float)
        seq_probs_2 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        expected_sequence_entropy = float(
            (
                -(seq_probs_1 * np.log(seq_probs_1)).sum()
                + -(seq_probs_2 * np.log(seq_probs_2)).sum()
            )
            / 2,
        )

        corpus_probs = np.array([3 / 6, 2 / 6, 1 / 6], dtype=float)
        expected_corpus_entropy = float(-(corpus_probs * np.log(corpus_probs)).sum())

        self.assertAlmostEqual(grouped["empirical_entropy"], expected_sequence_entropy)
        self.assertAlmostEqual(corpus["batch_empirical_entropy"], expected_corpus_entropy)
        self.assertNotAlmostEqual(grouped["empirical_entropy"], corpus["batch_empirical_entropy"])

    def test_evaluator_summary_includes_batch_empirical_entropy(self):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.perplexity_model = lambda texts, batch_size=0: {  # noqa: ARG005
            "perplexity": 2.0,
            "perplexity_ci95_lower": 1.5,
            "perplexity_ci95_upper": 2.5,
            "corpus_perplexity": 1.8,
        }
        evaluator.cosine_model = lambda texts: {  # noqa: ARG005
            "cosine_similarity": 0.4,
            "cosine_similarity_ci95": 0.05,
        }
        evaluator.wasserstein_model = None
        evaluator.string_metrics = type(
            "_FakeStringMetrics",
            (),
            {
                "reference_alignment": staticmethod(lambda predictions, references=None, num_workers=1: {}),  # noqa: ARG005
                "diversity_grouped": staticmethod(
                    lambda predictions, references=None, num_workers=1: {  # noqa: ARG005
                        "distinct_2": 0.3,
                        "distinct_2_ci95": 0.02,
                        "empirical_entropy": 0.7,
                        "empirical_entropy_ci95": 0.04,
                        "self_bleu": 12.0,
                        "self_bleu_ci95": 1.5,
                    },
                ),
                "diversity_corpus": staticmethod(
                    lambda predictions, references=None, prefix="batch": {  # noqa: ARG005
                        f"{prefix}_empirical_entropy": 0.9,
                    },
                ),
            },
        )()
        evaluator.batch_size = 0
        evaluator.force = False
        evaluator.show_timings = False

        metrics = evaluator.evaluate([["sample a", "sample b"]])
        summary = metrics.get("metrics_summary")

        self.assertIsInstance(summary, str)
        assert isinstance(summary, str)
        self.assertIn("Ent: 0.7 pm 0.04", summary)
        self.assertIn("B-Ent: 0.9", summary)

    def test_perplexity_empty_texts_are_penalized(self):
        perplexity = Perplexity.__new__(Perplexity)
        torch.nn.Module.__init__(perplexity)
        perplexity.model = _FakePerplexityModel()
        perplexity.tokenizer = _FakePerplexityTokenizer()
        perplexity.lm_head = torch.nn.Identity()

        scores = perplexity._forward(["non-empty", "   ", ""])

        self.assertIsNotNone(scores)
        assert scores is not None
        mean_nll = scores["mean_nll"].cpu().tolist()
        self.assertLess(mean_nll[0], 15.0)
        self.assertEqual(mean_nll[1], 15.0)
        self.assertEqual(mean_nll[2], 15.0)

    def test_evaluate_baseline_f1_requires_aligned_references(self):
        evaluator = Evaluator.__new__(Evaluator)

        with self.assertRaisesRegex(ValueError, "reference group per candidate group"):
            evaluator.evaluate_baseline(
                [["candidate-a"], ["candidate-b"]],
                metric="f1",
                k=1,
                references=[["ref-a"]],
            )

    def test_reference_alignment_requires_aligned_references(self):
        metrics = StringMetrics()

        with self.assertRaisesRegex(ValueError, "reference group per prediction group"):
            metrics.reference_alignment(
                [["candidate-a"], ["candidate-b"]],
                references=[["ref-a"]],
            )

    def test_oversample_baseline_evaluates_selected_groups_directly(self):
        class _RecordingEvaluator:
            def __init__(self):
                self.evaluate_inputs = None
                self.evaluate_references = None

            def evaluate_baseline(self, texts, metric, k, references=None):
                del texts, metric, k, references
                return [["a1", "a2"], ["b1", "b2"]]

            def evaluate(self, texts, references=None):
                self.evaluate_inputs = texts
                self.evaluate_references = references
                return {"f1": 1.0}

        evaluator = _RecordingEvaluator()
        references = [["ref-a"], ["ref-b"]]

        selected, metrics = _select_and_evaluate_baseline(
            evaluator,
            [["ignored-a"], ["ignored-b"]],
            metric="f1",
            subsample_k=2,
            references=references,
        )

        self.assertEqual(selected, [["a1", "a2"], ["b1", "b2"]])
        self.assertEqual(metrics, {"f1": 1.0})
        self.assertEqual(evaluator.evaluate_inputs, selected)
        self.assertEqual(evaluator.evaluate_references, references)

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

            self.assertTrue(_is_math_results_file(file_path))

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
            self.assertIn("empirical_entropy", math_metrics)
            self.assertIn("batch_distinct_2", math_metrics)
            self.assertIn("batch_empirical_entropy", math_metrics)
            self.assertIn("batch_self_bleu", math_metrics)
            self.assertIn("math_metrics_summary", math_metrics)

            with open(file_path) as f:
                saved = json.load(f)

            self.assertIn("math_metrics", saved)
            self.assertEqual(saved["math_metrics"]["batch_distinct_2"], math_metrics["batch_distinct_2"])


if __name__ == "__main__":
    unittest.main()

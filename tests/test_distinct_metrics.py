import os
import sys
import unittest


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from eval_core import StringMetrics


class TestStringMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = StringMetrics()

    def test_distinct_metrics_basic(self):
        texts = ["hello world", "hello python"]
        # tokens: [hello, world], [hello, python]
        # distinct_1: hello, world, python (3) / 4 = 0.75

        # 2grams: (hello, world), (hello, python) -> distinct: 2 / 2 = 1.0 (with nltk padding? let's check)
        # Wait, my implementation uses padding: pad_left=True, left_pad_symbol="<s>"
        # "hello world" -> (<s>, hello), (hello, world)
        # "hello python" -> (<s>, hello), (hello, python)
        # distinct 2grams: (<s>, hello), (hello, world), (hello, python) -> 3
        # total 2grams: 2 + 2 = 4
        # ratio: 3/4 = 0.75

        metrics = self.metrics.compute_distinct_metrics(texts)

        print("Metrics:", metrics)
        self.assertIn("distinct_1", metrics)
        self.assertIn("distinct_2", metrics)
        self.assertAlmostEqual(metrics["distinct_1"], 0.75)
        self.assertAlmostEqual(metrics["distinct_2"], 0.75)

    def test_ead_with_refs(self):
        texts = ["hello world", "hello python"]
        refs = ["hello world", "python code", "java code"]

        # Vocab should be calculated from refs
        # hello, world, python, code, java

        metrics = self.metrics.compute_distinct_metrics(texts, references_for_vocab=refs)
        print("EAD Metrics:", metrics)

        self.assertIn("expectation_adjusted_distinct", metrics)
        self.assertGreater(metrics["expectation_adjusted_distinct"], 0.0)

    def test_empty(self):
        metrics = self.metrics.compute_distinct_metrics([])
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()

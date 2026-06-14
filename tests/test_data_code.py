import os
import sys
import unittest
from unittest.mock import patch


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset


class _FakeSplit(list):
    def shuffle(self, seed: int):
        del seed
        return _FakeSplit(self)


class TestCodeDatasets(unittest.TestCase):
    @patch("d5p4.data.code_ds.load_dataset")
    def test_humaneval_normalizes_columns(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "test": _FakeSplit(
                [
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def add(a, b):\n",
                        "canonical_solution": "    return a + b\n",
                        "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
                        "entry_point": "add",
                    },
                ],
            ),
        }

        cfg = Config(disable_sys_args=True, code_dataset="humaneval")
        df = get_code_dataset(cfg)

        self.assertEqual(
            df.to_dict("records"),
            [
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n",
                    "reference_code": "    return a + b\n",
                    "tests": ["def check(candidate):\n    assert candidate(1, 2) == 3"],
                    "entry_point": "add",
                    "dataset": "humaneval",
                },
            ],
        )
        mock_load_dataset.assert_called_once_with("openai/openai_humaneval", cache_dir="./.cache")

    @patch("d5p4.data.code_ds.load_dataset")
    def test_mbpp_normalizes_sanitized_columns(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "test": _FakeSplit(
                [
                    {
                        "task_id": 2,
                        "prompt": "Write a function to add two numbers.",
                        "code": "def add(a, b):\n    return a + b",
                        "test_imports": ["import math"],
                        "test_list": ["assert add(1, 2) == 3"],
                    },
                ],
            ),
        }

        cfg = Config(disable_sys_args=True, code_dataset="mbpp")
        df = get_code_dataset(cfg)

        self.assertEqual(
            df.to_dict("records"),
            [
                {
                    "task_id": "2",
                    "prompt": "Write a function to add two numbers.",
                    "reference_code": "def add(a, b):\n    return a + b",
                    "tests": ["import math", "assert add(1, 2) == 3"],
                    "entry_point": "",
                    "dataset": "mbpp",
                },
            ],
        )
        mock_load_dataset.assert_called_once_with(
            "google-research-datasets/mbpp",
            "sanitized",
            cache_dir="./.cache",
        )

    @patch("d5p4.data.code_ds.load_dataset")
    def test_mbpp_formats_few_shot_prefix_from_train_split(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "train": _FakeSplit(
                [
                    {
                        "task_id": 10,
                        "prompt": "Write a function to double a number.",
                        "code": "def double(x):\n    return 2 * x",
                        "test_imports": [],
                        "test_list": ["assert double(3) == 6"],
                    },
                ],
            ),
            "test": _FakeSplit(
                [
                    {
                        "task_id": 2,
                        "prompt": "Write a function to add two numbers.",
                        "code": "def add(a, b):\n    return a + b",
                        "test_imports": [],
                        "test_list": ["assert add(1, 2) == 3"],
                    },
                ],
            ),
        }

        cfg = Config(disable_sys_args=True, code_dataset="mbpp", code_n_shots=1)
        df = get_code_dataset(cfg)

        self.assertIn("Problem: Write a function to double a number.", df.loc[0, "prompt"])
        self.assertIn("def double(x):", df.loc[0, "prompt"])
        self.assertTrue(
            df.loc[0, "prompt"].endswith("Problem: Write a function to add two numbers.\nSolution:\n```python\n"),
        )


if __name__ == "__main__":
    unittest.main()

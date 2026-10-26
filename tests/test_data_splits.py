import os
import sys
from unittest.mock import patch

import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset
from d5p4.data.math_ds import gsm8k
from d5p4.data.qa import get_qa_dataset


class _FakeSplit(list):
    shuffle_seeds: list[int] = []

    def shuffle(self, seed: int):
        _FakeSplit.shuffle_seeds.append(seed)
        return _FakeSplit(self)


def _gsm8k_item(question: str) -> dict:
    return {"question": question, "answer": "work\n#### 7"}


def _arc_item(question: str) -> dict:
    return {
        "question": question,
        "choices": {"text": ["yes", "no"], "label": ["A", "B"]},
        "answerKey": "A",
    }


@pytest.fixture(autouse=True)
def _reset_shuffle_seeds():
    _FakeSplit.shuffle_seeds = []


@patch("d5p4.data.math_ds.load_dataset")
def test_gsm8k_dev_uses_train_split(mock_load_dataset):
    mock_load_dataset.return_value = {
        "train": _FakeSplit([_gsm8k_item("train q")]),
        "test": _FakeSplit([_gsm8k_item("test q")]),
    }

    df = gsm8k(Config(disable_sys_args=True, qa_dataset="gsm8k", dataset_split="dev"))

    assert df["question"].tolist() == ["Question: train q\nAnswer:"]


@patch("d5p4.data.math_ds.load_dataset")
def test_gsm8k_eval_keeps_test_split(mock_load_dataset):
    mock_load_dataset.return_value = {
        "train": _FakeSplit([_gsm8k_item("train q")]),
        "test": _FakeSplit([_gsm8k_item("test q")]),
    }

    df = gsm8k(Config(disable_sys_args=True, qa_dataset="gsm8k"))

    assert df["question"].tolist() == ["Question: test q\nAnswer:"]


@patch("d5p4.data.math_ds.load_dataset")
def test_gsm8k_dev_rejects_few_shot_examples_from_the_same_split(mock_load_dataset):
    mock_load_dataset.return_value = {"train": _FakeSplit([_gsm8k_item("q")]), "test": _FakeSplit([])}

    with pytest.raises(ValueError, match="few-shot"):
        gsm8k(Config(disable_sys_args=True, qa_dataset="gsm8k", dataset_split="dev", qa_n_shots=4))


@patch("d5p4.data.math_ds.load_dataset")
def test_dataset_shuffle_seed_overrides_run_seed(mock_load_dataset):
    mock_load_dataset.return_value = {"train": _FakeSplit([]), "test": _FakeSplit([_gsm8k_item("q")])}

    gsm8k(Config(disable_sys_args=True, qa_dataset="gsm8k", seed=3))
    gsm8k(Config(disable_sys_args=True, qa_dataset="gsm8k", seed=3, dataset_shuffle_seed=0))

    assert _FakeSplit.shuffle_seeds == [3, 0]


@patch("d5p4.data.qa.load_dataset")
def test_ai2_arc_dev_uses_validation_split(mock_load_dataset):
    mock_load_dataset.return_value = {
        "train": _FakeSplit([]),
        "validation": _FakeSplit([_arc_item("validation q")]),
        "test": _FakeSplit([_arc_item("test q")]),
    }

    df = get_qa_dataset(Config(disable_sys_args=True, qa_dataset="ai2_arc", dataset_split="dev"))

    assert df["question"].tolist() == ["validation q"]


def test_truthful_qa_has_no_dev_split():
    with pytest.raises(ValueError, match="no split disjoint"):
        get_qa_dataset(Config(disable_sys_args=True, qa_dataset="truthful_qa", dataset_split="dev"))


def test_humaneval_has_no_dev_split():
    with pytest.raises(ValueError, match="no split disjoint"):
        get_code_dataset(Config(disable_sys_args=True, code_dataset="humaneval", dataset_split="dev"))


@patch("d5p4.data.code_ds.load_dataset")
def test_mbpp_dev_uses_full_validation_split(mock_load_dataset):
    sanitized = {
        "train": _FakeSplit(
            [
                {
                    "task_id": 601,
                    "prompt": "Double a number.",
                    "code": "def double(x):\n    return 2 * x",
                    "test_imports": [],
                    "test_list": ["assert double(3) == 6"],
                },
            ],
        ),
        "test": _FakeSplit([]),
    }
    full = {
        "validation": _FakeSplit(
            [
                {
                    "task_id": 511,
                    "text": "Add two numbers.",
                    "code": "def add(a, b):\n    return a + b",
                    "test_setup_code": "import math",
                    "test_list": ["assert add(1, 2) == 3"],
                },
            ],
        ),
    }
    mock_load_dataset.side_effect = lambda path, subset, cache_dir: sanitized if subset == "sanitized" else full

    df = get_code_dataset(
        Config(disable_sys_args=True, code_dataset="mbpp", code_n_shots=1, dataset_split="dev"),
    )

    record = df.to_dict("records")[0]
    assert record["task_id"] == "511"
    assert record["tests"] == ["import math", "assert add(1, 2) == 3"]
    assert record["prompt"].startswith('"""\nDouble a number.')
    assert record["prompt"].endswith('"""\nAdd two numbers.\nimport math\n"""')
    assert [call.args[1] for call in mock_load_dataset.call_args_list] == ["sanitized", "full"]

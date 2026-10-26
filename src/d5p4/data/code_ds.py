"""Code generation benchmark datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pandas as pd
from datasets import load_dataset

from d5p4.config import Config
from d5p4.data.splits import shuffle_seed, split_name


CODE_DATASETS = {"humaneval", "mbpp"}
# The sanitized MBPP config only has 43 validation tasks; the dev split uses the full config's 90.
MBPP_DEV_SUBSET = "full"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _limit(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.code_dataset_len > 0:
        return df.head(cfg.code_dataset_len)
    return df


def _format_mbpp_docstring(prompt_text: str, test_list: list[str] | None) -> str:
    first_test = test_list[0] if test_list else ""
    return f'"""\n{prompt_text}\n{first_test}\n"""'


def _format_mbpp_few_shot_prefix(examples: list[Any]) -> str:
    prefix = ""
    for item in examples:
        test_imports = _as_list(item.get("test_imports"))
        test_list = test_imports + _as_list(item.get("test_list"))
        docstring = _format_mbpp_docstring(item["prompt"], test_list)
        prefix += f"{docstring}\n{item['code']}\n\n"
    return prefix


def humaneval(cfg: Config) -> pd.DataFrame:
    """Load HumanEval test tasks with a normalized code-eval schema."""
    assert cfg.code_n_shots == 0, "HumanEval paper evaluation uses 0-shot prompting"
    split = split_name(cfg, "humaneval")
    dataset = cast(Any, load_dataset(cfg.humaneval_path, cache_dir=cfg.cache_dir)[split])
    dataset = dataset.shuffle(seed=shuffle_seed(cfg))

    rows = []
    for item in dataset:
        rows.append(
            {
                "task_id": str(item["task_id"]),
                "prompt": str(item["prompt"]),
                "reference_code": str(item["canonical_solution"]),
                "tests": [str(item["test"])],
                "entry_point": str(item["entry_point"]),
                "dataset": "humaneval",
            },
        )

    return _limit(pd.DataFrame(rows), cfg)


def mbpp(cfg: Config) -> pd.DataFrame:
    """Load MBPP tasks with a normalized code-eval schema.

    The eval split is the sanitized test split. The dev split is the full config's validation split; few-shot
    examples always come from the sanitized train split, which is disjoint from both.
    """
    split = split_name(cfg, "mbpp", cfg.code_n_shots)
    dataset_splits = cast(Any, load_dataset(cfg.mbpp_path, cfg.mbpp_subset, cache_dir=cfg.cache_dir))
    if cfg.dataset_split == "dev":
        dataset = cast(Any, load_dataset(cfg.mbpp_path, MBPP_DEV_SUBSET, cache_dir=cfg.cache_dir)[split])
    else:
        dataset = cast(Any, dataset_splits[split])
    dataset = dataset.shuffle(seed=shuffle_seed(cfg))
    train_dataset = cast(Any, dataset_splits["train"].shuffle(seed=shuffle_seed(cfg))) if cfg.code_n_shots > 0 else None

    rows = []
    for i, item in enumerate(dataset):
        # The full config names the prompt "text" and has setup code instead of test imports.
        test_imports = _as_list(item.get("test_imports")) + _as_list(item.get("test_setup_code"))
        tests = [*test_imports, *_as_list(item.get("test_list"))]
        prompt_text = str(item["prompt"] if "prompt" in item else item["text"])
        prompt = _format_mbpp_docstring(prompt_text, tests)
        if cfg.code_n_shots > 0:
            assert train_dataset is not None
            start_idx = i * cfg.code_n_shots
            examples: list[Any] = [
                train_dataset[idx % len(train_dataset)] for idx in range(start_idx, start_idx + cfg.code_n_shots)
            ]
            prompt = f"{_format_mbpp_few_shot_prefix(examples)}{prompt}"
        rows.append(
            {
                "task_id": str(item["task_id"]),
                "prompt": prompt,
                "reference_code": str(item["code"]),
                "tests": tests,
                "entry_point": "",
                "dataset": "mbpp",
            },
        )

    return _limit(pd.DataFrame(rows), cfg)


def get_code_dataset(cfg: Config) -> pd.DataFrame:
    """Get the configured code-generation benchmark dataset."""
    if cfg.code_dataset == "humaneval":
        return humaneval(cfg)
    if cfg.code_dataset == "mbpp":
        return mbpp(cfg)
    raise ValueError(f"Unknown code_dataset: {cfg.code_dataset}. Available: {sorted(CODE_DATASETS)}")

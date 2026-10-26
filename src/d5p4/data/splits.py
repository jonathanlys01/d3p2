"""Benchmark split selection for evaluation and hyperparameter tuning.

`dataset_split="eval"` keeps the split each benchmark reports results on. `dataset_split="dev"` selects a tuning
split that shares no questions with it, so hyperparameters picked on dev can be applied to eval without leakage.
"""

from __future__ import annotations

from d5p4.config import Config


EVAL_SPLITS = {
    "gsm8k": "test",
    "truthful_qa": "validation",
    "commonsense_qa": "validation",
    "ai2_arc": "test",
    "humaneval": "test",
    "mbpp": "test",
}

DEV_SPLITS = {
    "gsm8k": "train",
    # The CommonsenseQA test split has no labels, so validation is the evaluation split.
    "commonsense_qa": "train",
    "ai2_arc": "validation",
    # The full MBPP config's validation split (task ids 511-600) contains the sanitized validation tasks and is
    # disjoint from the sanitized test and train splits. See `d5p4.data.code_ds.mbpp`.
    "mbpp": "validation",
}

# Dev splits drawn from the same split as the few-shot examples.
_DEV_SHARES_FEW_SHOT_SOURCE = {"gsm8k", "commonsense_qa"}


def shuffle_seed(cfg: Config) -> int:
    """Seed for question order; a fixed `dataset_shuffle_seed` keeps subsets identical across run seeds."""
    return cfg.dataset_shuffle_seed if cfg.dataset_shuffle_seed >= 0 else cfg.seed


def split_name(cfg: Config, dataset_name: str, n_shots: int = 0) -> str:
    """Return the Hugging Face split that `cfg.dataset_split` selects for `dataset_name`."""
    if cfg.dataset_split == "eval":
        return EVAL_SPLITS[dataset_name]
    if dataset_name not in DEV_SPLITS:
        raise ValueError(
            f"{dataset_name} has no split disjoint from its evaluation split; tune on another benchmark "
            f"(available dev splits: {sorted(DEV_SPLITS)}).",
        )
    if n_shots > 0 and dataset_name in _DEV_SHARES_FEW_SHOT_SOURCE:
        raise ValueError(
            f"The {dataset_name} dev split is the train split that few-shot examples come from; use n_shots=0.",
        )
    return DEV_SPLITS[dataset_name]

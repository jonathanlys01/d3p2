"""Reproduce the January 2026 LLaDA/TruthfulQA score baseline.

This runner intentionally preserves the statistic used by
``likelihood_llada.py`` and the saved ``score_method_results_llada.npz`` files:

* the prompt and first correct answer are tokenized separately without a chat
  template or special tokens;
* the prompt is always visible and only answer tokens are masked;
* every batch of 16 contains a cyclic sweep over answer-relative mask counts;
* logits are scored over every answer position;
* negative entropy and uniform cross-entropy are min-max normalized across the
  16 mask draws before their batch mean is taken;
* four batch means (64 draws) are averaged into one score per TruthfulQA item;
* quality is conditional Llama mean log-likelihood of the answer.

The within-item min-max normalization is deliberately retained even though it
does not define an absolute score across items. This script is a historical
control, not the recommended fixed-mask-ratio experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase

from d5p4.config import Config
from d5p4.data.qa import get_qa_dataset
from d5p4.llada_ref.modeling_llada import LLaDAModelLM
from d5p4.utils import configure_runtime, process_model_args, seed_all, tqdm


LEGACY_MC_SAMPLES = 64
LEGACY_BATCH_SIZE = 16
LEGACY_MASK_TOKEN_ID = 126336


@dataclass(frozen=True)
class LegacySettings:
    """Script-only settings kept outside the global experiment Config."""

    num_items: int = -1
    output_prefix: str = "legacy_score_method_results_llada"

    def validate(self) -> None:
        if self.num_items == 0 or self.num_items < -1:
            raise ValueError("--legacy-num-items must be -1 or a positive integer")
        if not self.output_prefix.strip():
            raise ValueError("--legacy-output-prefix must not be blank")


def parse_legacy_settings(argv: list[str]) -> tuple[LegacySettings, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--legacy-num-items", type=int, default=LegacySettings.num_items)
    parser.add_argument("--legacy-output-prefix", default=LegacySettings.output_prefix)
    parsed, remaining = parser.parse_known_args(argv)
    settings = LegacySettings(
        num_items=parsed.legacy_num_items,
        output_prefix=parsed.legacy_output_prefix,
    )
    settings.validate()
    return settings, remaining


def config_from_remaining_args(remaining: list[str]) -> Config:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        return Config()
    finally:
        sys.argv = original_argv


def legacy_mask_batch(
    batch: torch.Tensor,
    *,
    prompt_length: int,
    mask_token_id: int = LEGACY_MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the historical cyclic mask-count sweep to answer tokens."""
    batch_size, sequence_length = batch.shape
    answer_length = sequence_length - prompt_length
    if answer_length <= 0:
        raise ValueError("Legacy masking requires at least one answer token")

    start_count = torch.randint(1, answer_length + 1, (), device=batch.device)
    mask_counts = torch.round(
        torch.linspace(
            float(start_count),
            start_count + (batch_size - 1) * (answer_length / batch_size),
            steps=batch_size,
            device=batch.device,
        ),
    ).long()
    mask_counts = ((mask_counts - 1) % answer_length) + 1

    answer_mask = torch.arange(answer_length, device=batch.device).repeat(batch_size, 1)
    answer_mask = answer_mask < mask_counts.unsqueeze(1)
    for row in range(batch_size):
        # The January runner created this permutation on CPU, even for a CUDA mask.
        answer_mask[row] = answer_mask[row][torch.randperm(answer_length)]

    full_mask = torch.cat(
        [
            torch.zeros((batch_size, prompt_length), dtype=torch.bool, device=batch.device),
            answer_mask,
        ],
        dim=1,
    )
    return torch.where(full_mask, mask_token_id, batch), mask_counts


def legacy_scores_from_log_probs(log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return historical per-draw scores, including batch min-max scaling."""
    probabilities = log_probs.float().exp()
    entropy = -(probabilities * log_probs.float()).sum(dim=-1)
    entropy_raw = -entropy.mean(dim=-1)
    self_certainty_raw = -log_probs.float().mean(dim=-1).mean(dim=-1)

    def normalize(values: torch.Tensor) -> torch.Tensor:
        return (values - values.min()) / (values.max() - values.min() + 1e-12)

    return normalize(entropy_raw), normalize(self_certainty_raw)


def _repeated_sequence(prompt: torch.Tensor, answer: torch.Tensor) -> tuple[torch.Tensor, int]:
    sequence = torch.cat([prompt, answer]).unsqueeze(0)
    return sequence.repeat(LEGACY_BATCH_SIZE, 1), int(prompt.numel())


def advance_discarded_likelihood_masks(prompt: torch.Tensor, answer: torch.Tensor) -> None:
    """Consume the masks drawn by the historical LLaDA-likelihood calculation.

    The old runner estimated LLaDA likelihood immediately before its internal
    scores. Its eval-mode model forwards do not draw random numbers, so drawing
    and discarding these four mask batches preserves the score-mask RNG stream
    without doubling model inference cost.
    """
    repeated, prompt_length = _repeated_sequence(prompt, answer)
    for _ in range(LEGACY_MC_SAMPLES // LEGACY_BATCH_SIZE):
        legacy_mask_batch(repeated, prompt_length=prompt_length)


@torch.inference_mode()
def compute_legacy_internal_scores(
    model: LLaDAModelLM,
    prompt: torch.Tensor,
    answer: torch.Tensor,
) -> tuple[float, float]:
    repeated, prompt_length = _repeated_sequence(prompt, answer)
    entropy_batch_means: list[float] = []
    self_batch_means: list[float] = []

    for _ in range(LEGACY_MC_SAMPLES // LEGACY_BATCH_SIZE):
        masked, _ = legacy_mask_batch(repeated, prompt_length=prompt_length)
        output = model(masked, return_dict=True)
        log_probs = F.log_softmax(output.logits[:, prompt_length:, :], dim=-1)
        entropy, self_certainty = legacy_scores_from_log_probs(log_probs)
        entropy_batch_means.append(float(entropy.mean().item()))
        self_batch_means.append(float(self_certainty.mean().item()))

    return float(np.mean(entropy_batch_means)), float(np.mean(self_batch_means))


@torch.inference_mode()
def compute_legacy_ar_likelihood(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    answer_text: str,
) -> float:
    """Return the historical conditional mean log-likelihood ``-loss``."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids.to(model.device)
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return -float(model(input_ids, labels=labels).loss.item())


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _signature(config: Config, rows: list[tuple[int, str, str]]) -> str:
    payload = {
        "version": 1,
        "statistic": "january_2026_mixed_masks_answer_logits_batch_minmax",
        "seed": config.seed,
        "llada_model_path": config.llada_model_path,
        "ar_model_path": config.ar_model_path,
        "mc_samples": LEGACY_MC_SAMPLES,
        "batch_size": LEGACY_BATCH_SIZE,
        "items": [
            [index, hashlib.sha256(question.encode()).hexdigest(), hashlib.sha256(answer.encode()).hexdigest()]
            for index, question, answer in rows
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_existing_points(path: Path, signature: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    points = pd.read_csv(path)
    if "experiment_signature" not in points.columns:
        raise RuntimeError(f"Existing legacy points at {path} have no experiment signature")
    signatures = set(points["experiment_signature"].dropna().astype(str))
    if signatures != {signature}:
        raise RuntimeError(f"Existing legacy points at {path} have a different experiment signature")
    if points["dataset_index"].duplicated().any():
        raise RuntimeError(f"Existing legacy points at {path} contain duplicate dataset indices")
    return points


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    entropy = spearmanr(points["entropy_score"], points["ar_mean_log_likelihood"])
    self_certainty = spearmanr(points["self_certainty_score"], points["ar_mean_log_likelihood"])
    return pd.DataFrame(
        [
            {
                "model": "llada",
                "dataset": "truthful_qa",
                "n_items": len(points),
                "mc_samples": LEGACY_MC_SAMPLES,
                "batch_size": LEGACY_BATCH_SIZE,
                "masking": "mixed_answer_relative_1_to_100pct",
                "score_scope": "all_answer_positions",
                "normalization": "within_item_per_batch_minmax",
                "entropy_spearman_rho_vs_ar_ll": float(entropy.statistic),
                "entropy_p_value": float(entropy.pvalue),
                "self_certainty_spearman_rho_vs_ar_ll": float(self_certainty.statistic),
                "self_certainty_p_value": float(self_certainty.pvalue),
                "entropy_advantage": float(entropy.statistic - self_certainty.statistic),
            },
        ],
    )


def _load_models(
    config: Config,
    device: torch.device,
) -> tuple[LLaDAModelLM, PreTrainedTokenizerBase, LlamaForCausalLM, PreTrainedTokenizerBase]:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    llada = LLaDAModelLM.from_pretrained(
        **process_model_args(
            config.llada_model_path,
            trust_remote_code=True,
            dtype=dtype,
            cache_dir=config.cache_dir,
        ),
    ).to(device).eval()
    llada_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            **process_model_args(
                config.llada_model_path,
                trust_remote_code=True,
                cache_dir=config.cache_dir,
            ),
        ),
    )
    ar_model = LlamaForCausalLM.from_pretrained(
        **process_model_args(
            config.ar_model_path,
            dtype=dtype,
            cache_dir=config.cache_dir,
        ),
    ).to(device).eval()
    ar_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            **process_model_args(config.ar_model_path, cache_dir=config.cache_dir),
        ),
    )
    return llada, llada_tokenizer, ar_model, ar_tokenizer


def main() -> None:  # noqa: PLR0915
    settings, remaining = parse_legacy_settings(sys.argv[1:])
    config = config_from_remaining_args(remaining)
    configure_runtime(config)

    dataset_config = replace(
        config,
        disable_sys_args=True,
        qa_dataset="truthful_qa",
        qa_dataset_len=-1,
        qa_n_shots=0,
    )
    frame = get_qa_dataset(dataset_config).reset_index(drop=True)
    source_rows = [
        (index, str(row["question"]), str(row["correct_answers"][0]))
        for index, (_, row) in enumerate(frame.iterrows())
        if len(row["correct_answers"]) > 0
    ]
    rows = source_rows if settings.num_items < 0 else source_rows[: settings.num_items]
    if len(rows) < 2:
        raise RuntimeError("The legacy correlation needs at least two TruthfulQA items")

    signature = _signature(config, rows)
    output_root = Path(config.results_dir)
    points_path = output_root / f"{settings.output_prefix}_points.csv"
    summary_path = output_root / f"{settings.output_prefix}_correlations.csv"
    npz_path = output_root / f"{settings.output_prefix}.npz"
    existing = _load_existing_points(points_path, signature)
    point_rows = existing.to_dict("records") if not existing.empty else []
    completed = {int(row["dataset_index"]) for row in point_rows}

    print("Legacy January 2026 score baseline")
    print(f"  TruthfulQA items: {len(rows)}")
    print("  masks: four mixed-ratio batches of 16; prompt always visible")
    print("  scores: all answer logits, within-item per-batch min-max normalization")
    print("  reference: conditional Llama answer mean log-likelihood")

    if len(completed) < len(rows):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_all(config.seed)
        llada, llada_tokenizer, ar_model, ar_tokenizer = _load_models(config, device)

        pending_since_flush = 0
        for dataset_index, question, answer in tqdm(rows, desc="Legacy score baseline"):
            prompt_ids = llada_tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_ids = llada_tokenizer(answer, add_special_tokens=False)["input_ids"]
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)
            if answer_tensor.numel() == 0:
                raise RuntimeError(f"TruthfulQA row {dataset_index} has an empty tokenized answer")

            # Advance both historical mask streams even when resuming a completed row.
            advance_discarded_likelihood_masks(prompt, answer_tensor)
            if dataset_index in completed:
                repeated, prompt_length = _repeated_sequence(prompt, answer_tensor)
                for _ in range(LEGACY_MC_SAMPLES // LEGACY_BATCH_SIZE):
                    legacy_mask_batch(repeated, prompt_length=prompt_length)
                continue

            entropy_score, self_certainty_score = compute_legacy_internal_scores(
                llada,
                prompt,
                answer_tensor,
            )
            ar_ll = compute_legacy_ar_likelihood(ar_model, ar_tokenizer, question, answer)
            point_rows.append(
                {
                    "experiment_signature": signature,
                    "dataset_index": dataset_index,
                    "question": question,
                    "answer": answer,
                    "answer_tokens": int(answer_tensor.numel()),
                    "entropy_score": entropy_score,
                    "self_certainty_score": self_certainty_score,
                    "ar_mean_log_likelihood": ar_ll,
                    "ar_ppl": math.exp(-ar_ll),
                },
            )
            completed.add(dataset_index)
            pending_since_flush += 1
            if pending_since_flush >= LEGACY_BATCH_SIZE:
                _atomic_write_csv(pd.DataFrame(point_rows), points_path)
                pending_since_flush = 0

    points = pd.DataFrame(point_rows).sort_values("dataset_index").reset_index(drop=True)
    if len(points) != len(rows):
        raise RuntimeError(f"Legacy point coverage mismatch: expected {len(rows)}, found {len(points)}")
    _atomic_write_csv(points, points_path)
    summary = summarize(points)
    _atomic_write_csv(summary, summary_path)
    np.savez(
        npz_path,
        entropy_scores=points["entropy_score"].to_numpy(),
        self_certainty_scores=points["self_certainty_score"].to_numpy(),
        ar_ll=points["ar_mean_log_likelihood"].to_numpy(),
    )

    print("\nLegacy correlation results")
    display_columns = [
        "n_items",
        "entropy_spearman_rho_vs_ar_ll",
        "self_certainty_spearman_rho_vs_ar_ll",
        "entropy_advantage",
    ]
    display = cast(pd.DataFrame, summary.loc[:, display_columns]).round(6)
    print(display.to_string(index=False))
    print(f"Saved points: {points_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved NPZ: {npz_path}")


if __name__ == "__main__":
    main()

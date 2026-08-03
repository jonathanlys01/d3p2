"""Compare partial-hypothesis certainty proxies with conditional Llama PPL.

The experiment treats a benchmark reference completion as a hypothesis, keeps
its prompt visible, and replaces an exact fraction of completion tokens with
the scorer's mask token.  It compares two properties of the raw conditional
token distribution at masked positions:

* entropy certainty: ``KL(p || uniform) / log(V)``;
* self-certainty: ``KL(uniform || p) / log(V)``.

Both are zero for a uniform distribution and increase as the distribution
becomes concentrated.  Entropy weights tokens according to the model's own
probability mass, whereas self-certainty weights every vocabulary token
equally and is therefore much more sensitive to tiny tail probabilities.

Sixteen independently masked versions of one item are evaluated as one batch.
Their scores are averaged before computing task-level Spearman correlations,
so the 128 benchmark items -- not the mask draws -- are the independent points.
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
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput

from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset
from d5p4.data.math_ds import gsm8k
from d5p4.data.qa import get_qa_dataset
from d5p4.dream_ref.modeling_dream import DreamModel
from d5p4.llada_ref.modeling_llada import LLaDAModelLM
from d5p4.utils import configure_runtime, get_tokenizer, process_model_args, tqdm


SUPPORTED_MODELS = ("llada", "dream")
SUPPORTED_DATASETS = ("truthful_qa", "ai2_arc", "gsm8k", "mbpp", "humaneval")
TASK_FAMILIES = {
    "truthful_qa": "qa",
    "ai2_arc": "qa",
    "gsm8k": "math",
    "mbpp": "code",
    "humaneval": "code",
}


@dataclass(frozen=True)
class ReferenceItem:
    dataset: str
    task_family: str
    item_id: str
    dataset_index: int
    prompt_text: str
    completion_text: str


@dataclass(frozen=True)
class EligibilityCounts:
    source_items: int
    eligible_items: int


@dataclass(frozen=True)
class ExperimentSettings:
    """Script-local knobs excluded from the global Config/resume identity."""

    models: str = "llada,dream"
    datasets: str = "truthful_qa,ai2_arc,gsm8k,mbpp,humaneval"
    num_items: int = 128
    mask_ratios: str = "0.15,0.50,0.85"
    mask_draws: int = 16
    batch_size: int = 16
    min_completion_tokens: int = 4
    bootstrap_samples: int = 2_000
    output_prefix: str = "partial_hypothesis_score"

    def validate(self) -> None:
        if self.num_items <= 1:
            raise ValueError("--score-correlation-num-items must be greater than one")
        if self.mask_draws < 2:
            raise ValueError("--score-correlation-mask-draws must be at least two")
        if not 0 < self.batch_size <= 16:
            raise ValueError("--score-correlation-batch-size must be in [1, 16]")
        if self.min_completion_tokens < 2:
            raise ValueError("--score-correlation-min-completion-tokens must be at least two")
        if self.bootstrap_samples <= 0:
            raise ValueError("--score-correlation-bootstrap-samples must be positive")
        if not self.output_prefix.strip():
            raise ValueError("--score-correlation-output-prefix must not be blank")


def parse_experiment_settings(argv: list[str]) -> tuple[ExperimentSettings, list[str]]:
    """Parse only this script's flags and leave OmegaConf/Config flags untouched."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--score-correlation-models", default=ExperimentSettings.models)
    parser.add_argument("--score-correlation-datasets", default=ExperimentSettings.datasets)
    parser.add_argument("--score-correlation-num-items", type=int, default=ExperimentSettings.num_items)
    parser.add_argument("--score-correlation-mask-ratios", default=ExperimentSettings.mask_ratios)
    parser.add_argument("--score-correlation-mask-draws", type=int, default=ExperimentSettings.mask_draws)
    parser.add_argument("--score-correlation-batch-size", type=int, default=ExperimentSettings.batch_size)
    parser.add_argument(
        "--score-correlation-min-completion-tokens",
        type=int,
        default=ExperimentSettings.min_completion_tokens,
    )
    parser.add_argument(
        "--score-correlation-bootstrap-samples",
        type=int,
        default=ExperimentSettings.bootstrap_samples,
    )
    parser.add_argument("--score-correlation-output-prefix", default=ExperimentSettings.output_prefix)
    parsed, remaining = parser.parse_known_args(argv)
    settings = ExperimentSettings(
        models=parsed.score_correlation_models,
        datasets=parsed.score_correlation_datasets,
        num_items=parsed.score_correlation_num_items,
        mask_ratios=parsed.score_correlation_mask_ratios,
        mask_draws=parsed.score_correlation_mask_draws,
        batch_size=parsed.score_correlation_batch_size,
        min_completion_tokens=parsed.score_correlation_min_completion_tokens,
        bootstrap_samples=parsed.score_correlation_bootstrap_samples,
        output_prefix=parsed.score_correlation_output_prefix,
    )
    settings.validate()
    return settings, remaining


def config_from_remaining_args(remaining: list[str]) -> Config:
    """Construct Config without exposing script-local flags to OmegaConf."""
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        return Config()
    finally:
        sys.argv = original_argv


def _parse_names(value: str, *, supported: tuple[str, ...], field_name: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError(f"{field_name} must contain at least one value")
    unknown = sorted(set(names) - set(supported))
    if unknown:
        raise ValueError(f"Unknown {field_name} values {unknown}; expected a subset of {list(supported)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} contains duplicate values: {names}")
    return names


def parse_mask_ratios(value: str) -> list[float]:
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not ratios:
        raise ValueError("score_correlation_mask_ratios must contain at least one ratio")
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"Mask ratios must be unique, got {ratios}")
    if any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise ValueError(f"Mask ratios must be strictly between zero and one, got {ratios}")
    return ratios


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF


def _native_prompt_text(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    *,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return prompt_text
    return cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        ),
    )


def tokenize_prompt_completion(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    completion_text: str,
    *,
    use_chat_template: bool,
    prompt_add_special_tokens: bool = False,
) -> tuple[torch.Tensor, int, int]:
    native_prompt = _native_prompt_text(tokenizer, prompt_text, use_chat_template=use_chat_template)
    prompt_ids = tokenizer(native_prompt, add_special_tokens=prompt_add_special_tokens)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Tokenized prompt is empty")
    if not completion_ids:
        raise ValueError("Tokenized completion is empty")
    ids = torch.tensor([*prompt_ids, *completion_ids], dtype=torch.long)
    return ids, len(prompt_ids), len(completion_ids)


def build_mask_draws(  # noqa: PLR0913
    input_ids: torch.Tensor,
    *,
    prompt_length: int,
    mask_token_id: int,
    mask_ratio: float,
    num_draws: int,
    seed_parts: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return exact-count masks over completion tokens only."""
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be one-dimensional, got {tuple(input_ids.shape)}")
    completion_length = input_ids.numel() - prompt_length
    if completion_length < 2:
        raise ValueError("A partial hypothesis needs at least two completion tokens")

    num_masked = min(completion_length - 1, max(1, int(round(mask_ratio * completion_length))))
    completion_masks = torch.zeros((num_draws, completion_length), dtype=torch.bool)
    for draw_index in range(num_draws):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(*seed_parts, draw_index))
        selected = torch.randperm(completion_length, generator=generator)[:num_masked]
        completion_masks[draw_index, selected] = True

    masked_ids = input_ids.unsqueeze(0).repeat(num_draws, 1)
    answer_view = masked_ids[:, prompt_length:]
    answer_view[completion_masks] = mask_token_id
    return masked_ids, completion_masks, num_masked


def certainty_scores_from_logits(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one entropy-certainty and self-certainty score per batch row."""
    if logits.shape[:-1] != masked_positions.shape:
        raise ValueError(
            f"Logit positions {tuple(logits.shape[:-1])} do not match mask {tuple(masked_positions.shape)}",
        )
    if not torch.all(masked_positions.sum(dim=1) > 0):
        raise ValueError("Every row must contain at least one masked position")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    log_vocab = math.log(logits.size(-1))

    entropy_certainty = 1.0 + (probs * log_probs).sum(dim=-1) / log_vocab
    self_certainty = -log_probs.mean(dim=-1) / log_vocab - 1.0

    weights = masked_positions.to(dtype=log_probs.dtype)
    denominator = weights.sum(dim=1)
    entropy_per_row = (entropy_certainty * weights).sum(dim=1) / denominator
    self_certainty_per_row = (self_certainty * weights).sum(dim=1) / denominator
    return entropy_per_row, self_certainty_per_row


def score_llada_mask_batch(
    model: LLaDAModelLM,
    masked_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    *,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score LLaDA logits aligned with the same completion positions."""
    with torch.inference_mode():
        output = model(
            cast(torch.LongTensor, masked_ids),
            return_dict=True,
            output_hidden_states=False,
            last_hidden_state_only=True,
            logits_slice=slice(prompt_length, None),
        )
    assert isinstance(output, CausalLMOutputWithPast) and output.logits is not None
    return certainty_scores_from_logits(output.logits, completion_masks)


def score_dream_mask_batch(
    model: DreamModel,
    masked_ids: torch.Tensor,
    completion_masks: torch.Tensor,
    *,
    completion_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score Dream's token-i distribution from predictor position i-1."""
    with torch.inference_mode():
        output = model(
            cast(torch.LongTensor, masked_ids),
            attention_mask=cast(Any, "full"),
            return_dict=True,
            output_hidden_states=False,
            use_cache=False,
            num_logits_to_keep=completion_length + 1,
        )
    assert isinstance(output, MaskedLMOutput) and output.logits is not None
    aligned_logits = output.logits[:, :-1]
    if aligned_logits.shape[1] != completion_length:
        raise ValueError(
            f"Dream returned {aligned_logits.shape[1]} aligned positions for {completion_length} completion tokens",
        )
    return certainty_scores_from_logits(aligned_logits, completion_masks)


def _reference_config(config: Config, **changes: Any) -> Config:
    return replace(config, disable_sys_args=True, qa_n_shots=0, code_n_shots=0, **changes)


def load_reference_items(config: Config, dataset_name: str) -> list[ReferenceItem]:
    if dataset_name in {"truthful_qa", "ai2_arc"}:
        cfg = _reference_config(config, qa_dataset=dataset_name, qa_dataset_len=-1)
        frame = get_qa_dataset(cfg)
        records = []
        for index, row in frame.reset_index(drop=True).iterrows():
            answers = list(row["correct_answers"])
            if answers:
                records.append((index, str(row["question"]), str(answers[0])))
    elif dataset_name == "gsm8k":
        cfg = _reference_config(config, model="llada", qa_dataset="gsm8k", qa_dataset_len=-1)
        frame = gsm8k(cfg)
        records = [
            (index, str(row["question"]), str(row["answer_str"]))
            for index, (_, row) in enumerate(frame.iterrows())
        ]
    elif dataset_name in {"mbpp", "humaneval"}:
        cfg = _reference_config(config, code_dataset=dataset_name, code_dataset_len=-1)
        frame = get_code_dataset(cfg)
        records = [
            (index, str(row["prompt"]), str(row["reference_code"]))
            for index, (_, row) in enumerate(frame.iterrows())
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return [
        ReferenceItem(
            dataset=dataset_name,
            task_family=TASK_FAMILIES[dataset_name],
            item_id=(
                str(frame.iloc[index]["task_id"])
                if dataset_name in {"mbpp", "humaneval"}
                else f"{dataset_name}:{index}"
            ),
            dataset_index=index,
            prompt_text=prompt,
            completion_text=completion,
        )
        for index, prompt, completion in records
        if prompt.strip() and completion.strip()
    ]


def select_shared_eligible_items(
    source_items: list[ReferenceItem],
    *,
    tokenizers: dict[str, PreTrainedTokenizerBase],
    config: Config,
    num_items: int,
    min_completion_tokens: int,
) -> tuple[list[ReferenceItem], EligibilityCounts]:
    selected: list[ReferenceItem] = []
    eligible_count = 0
    max_total_tokens = config.sequence_length

    for item in source_items:
        eligible = True
        for model_name in SUPPORTED_MODELS:
            tokenizer = tokenizers[model_name]
            use_chat = model_name == "dream" or (
                model_name == "llada" and "instruct" in config.llada_model_path.lower()
            )
            try:
                ids, _, completion_length = tokenize_prompt_completion(
                    tokenizer,
                    item.prompt_text,
                    item.completion_text,
                    use_chat_template=use_chat,
                )
            except ValueError:
                eligible = False
                break
            if completion_length < min_completion_tokens or ids.numel() > max_total_tokens:
                eligible = False
                break

        if eligible:
            eligible_count += 1
            if len(selected) < num_items:
                selected.append(item)

    counts = EligibilityCounts(source_items=len(source_items), eligible_items=eligible_count)
    if len(selected) < num_items:
        dataset = source_items[0].dataset if source_items else "unknown"
        raise RuntimeError(
            f"Dataset {dataset!r} supplied only {len(selected)} eligible shared references; "
            f"need {num_items}. Source={counts.source_items}, eligible={counts.eligible_items}, "
            f"minimum completion tokens={min_completion_tokens}, "
            f"maximum total tokens={max_total_tokens}.",
        )
    return selected, counts


def _pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for value in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
        if isinstance(value, int):
            return value
    raise ValueError("Llama tokenizer needs a pad, EOS, or BOS token id")


def compute_conditional_llama_ppl_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    items: list[ReferenceItem],
    *,
    device: torch.device,
    use_chat_template: bool,
) -> list[tuple[float, float, int]]:
    """Return completion mean NLL, PPL, and scored token count for each item."""
    encoded: list[tuple[list[int], list[int]]] = []
    for item in items:
        native_prompt = _native_prompt_text(
            tokenizer,
            item.prompt_text,
            use_chat_template=use_chat_template,
        )
        prompt_ids = tokenizer(
            native_prompt,
            add_special_tokens=not use_chat_template,
        )["input_ids"]
        completion_ids = tokenizer(item.completion_text, add_special_tokens=False)["input_ids"]
        if not prompt_ids or not completion_ids:
            raise ValueError(f"Llama tokenization produced an empty prompt/completion for {item.item_id}")
        encoded.append((list(prompt_ids), list(completion_ids)))

    max_length = max(len(prompt) + len(completion) for prompt, completion in encoded)
    pad_id = _pad_id(tokenizer)
    input_ids = torch.full((len(items), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (prompt_ids, completion_ids) in enumerate(encoded):
        sequence = torch.tensor([*prompt_ids, *completion_ids], dtype=torch.long, device=device)
        length = sequence.numel()
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
        labels[row, len(prompt_ids) : length] = sequence[len(prompt_ids) :]

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = output.logits.float()
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    results = []
    for row in range(len(items)):
        token_count = int(valid[row].sum().item())
        if token_count == 0:
            raise ValueError(f"No completion tokens were scored for {items[row].item_id}")
        mean_nll = float(losses[row][valid[row]].mean().item())
        ppl = math.exp(mean_nll) if mean_nll < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else float("inf")
        results.append((mean_nll, ppl, token_count))
    return results


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _experiment_signature(  # noqa: PLR0913
    config: Config,
    *,
    settings: ExperimentSettings,
    models: list[str],
    datasets: list[str],
    ratios: list[float],
    selected: dict[str, list[ReferenceItem]],
) -> str:
    payload = {
        "version": 1,
        "seed": config.seed,
        "models": models,
        "datasets": datasets,
        "ratios": ratios,
        "num_items": settings.num_items,
        "mask_draws": settings.mask_draws,
        "batch_size": settings.batch_size,
        "min_completion_tokens": settings.min_completion_tokens,
        "sequence_length": config.sequence_length,
        "llada_model_path": config.llada_model_path,
        "dream_model_path": config.dream_model_path,
        "ar_model_path": config.ar_model_path,
        "items": {
            dataset: [
                [
                    item.item_id,
                    hashlib.sha256(item.prompt_text.encode()).hexdigest(),
                    hashlib.sha256(item.completion_text.encode()).hexdigest(),
                ]
                for item in items
            ]
            for dataset, items in selected.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_resumable_points(path: Path, signature: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "experiment_signature" not in frame.columns:
        raise RuntimeError(f"Existing points file {path} has no experiment_signature; refusing to mix runs")
    signatures = set(frame["experiment_signature"].dropna().astype(str))
    if signatures != {signature}:
        raise RuntimeError(
            f"Existing points file {path} belongs to a different experiment signature: {sorted(signatures)}",
        )
    key_columns = ["model", "dataset", "item_id", "mask_ratio"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError(f"Existing points file {path} contains duplicate correlation-point keys")
    return frame


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return float("nan"), float("nan"), "insufficient_finite_points"
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan"), "constant_input"
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), "ok"


def _bootstrap_correlations(
    entropy: np.ndarray,
    self_certainty: np.ndarray,
    ppl: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(ppl)
    entropy_rhos: list[float] = []
    self_rhos: list[float] = []
    quality_deltas: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        entropy_rho, _, entropy_status = _safe_spearman(entropy[indices], ppl[indices])
        self_rho, _, self_status = _safe_spearman(self_certainty[indices], ppl[indices])
        if entropy_status == "ok":
            entropy_rhos.append(entropy_rho)
        if self_status == "ok":
            self_rhos.append(self_rho)
        if entropy_status == self_status == "ok":
            quality_deltas.append((-entropy_rho) - (-self_rho))

    def bounds(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        return float(low), float(high)

    entropy_low, entropy_high = bounds(entropy_rhos)
    self_low, self_high = bounds(self_rhos)
    delta_low, delta_high = bounds(quality_deltas)
    return {
        "entropy_spearman_rho_vs_ppl_ci95_low": entropy_low,
        "entropy_spearman_rho_vs_ppl_ci95_high": entropy_high,
        "entropy_quality_rho_ci95_low": -entropy_high,
        "entropy_quality_rho_ci95_high": -entropy_low,
        "self_certainty_spearman_rho_vs_ppl_ci95_low": self_low,
        "self_certainty_spearman_rho_vs_ppl_ci95_high": self_high,
        "self_certainty_quality_rho_ci95_low": -self_high,
        "self_certainty_quality_rho_ci95_high": -self_low,
        "entropy_quality_advantage_ci95_low": delta_low,
        "entropy_quality_advantage_ci95_high": delta_high,
    }


def summarize_correlations(
    points: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = points.groupby(["model", "dataset", "task_family", "mask_ratio"], sort=False)
    for group_key, group in cast(Any, grouped):
        model_name, dataset, task_family, mask_ratio = cast(tuple[str, str, str, float], group_key)
        entropy = group["entropy_certainty_mean"].to_numpy(dtype=float)
        self_certainty = group["self_certainty_mean"].to_numpy(dtype=float)
        ppl = group["llama_ppl"].to_numpy(dtype=float)
        entropy_rho, entropy_p, entropy_status = _safe_spearman(entropy, ppl)
        self_rho, self_p, self_status = _safe_spearman(self_certainty, ppl)
        bootstrap = _bootstrap_correlations(
            entropy,
            self_certainty,
            ppl,
            samples=bootstrap_samples,
            seed=_stable_seed(seed, model_name, dataset, mask_ratio, "bootstrap"),
        )
        status = "ok" if entropy_status == self_status == "ok" else f"entropy:{entropy_status};self:{self_status}"
        rows.append(
            {
                "experiment_signature": str(group["experiment_signature"].iloc[0]),
                "model": model_name,
                "dataset": dataset,
                "task_family": task_family,
                "mask_ratio": float(mask_ratio),
                "n_items": len(group),
                "mask_draws": int(group["mask_draws"].iloc[0]),
                "source_items": int(group["source_items"].iloc[0]),
                "eligible_items": int(group["eligible_items"].iloc[0]),
                "entropy_spearman_rho_vs_ppl": entropy_rho,
                "entropy_quality_rho": -entropy_rho,
                "entropy_spearman_p_value": entropy_p,
                "self_certainty_spearman_rho_vs_ppl": self_rho,
                "self_certainty_quality_rho": -self_rho,
                "self_certainty_spearman_p_value": self_p,
                "entropy_quality_advantage": (-entropy_rho) - (-self_rho),
                "mean_entropy_draw_sd": float(np.mean(group["entropy_certainty_sd"].to_numpy(dtype=float))),
                "mean_self_certainty_draw_sd": float(
                    np.mean(group["self_certainty_sd"].to_numpy(dtype=float)),
                ),
                "mean_realized_mask_ratio": float(
                    np.mean(group["realized_mask_ratio"].to_numpy(dtype=float)),
                ),
                "status": status,
                **bootstrap,
            },
        )
    return pd.DataFrame(rows)


def _load_llama(config: Config, device: torch.device) -> tuple[LlamaForCausalLM, PreTrainedTokenizerBase]:
    model_args = process_model_args(
        config.ar_model_path,
        cache_dir=config.cache_dir,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model = LlamaForCausalLM.from_pretrained(**model_args).to(device).eval()
    tokenizer_args = process_model_args(config.ar_tokenizer, cache_dir=config.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args, trust_remote_code=True)
    return model, cast(PreTrainedTokenizerBase, tokenizer)


def _load_scorer(config: Config, model_name: str, device: torch.device) -> torch.nn.Module:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if model_name == "llada":
        args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype=dtype)
        return LLaDAModelLM.from_pretrained(**args).to(device).eval()
    if model_name == "dream":
        args = process_model_args(config.dream_model_path, cache_dir=config.cache_dir, dtype=dtype)
        return DreamModel.from_pretrained(**args).to(device).eval()
    raise ValueError(f"Unsupported scorer model: {model_name}")


def _release_model(model: torch.nn.Module, device: torch.device) -> None:
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _llama_scores_for_items(
    config: Config,
    settings: ExperimentSettings,
    selected: dict[str, list[ReferenceItem]],
    *,
    device: torch.device,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    print(f"Loading Llama reference model from {config.ar_model_path}")
    model, tokenizer = _load_llama(config, device)
    use_chat = bool(getattr(tokenizer, "chat_template", None)) and "instruct" in config.ar_model_path.lower()
    scores: dict[tuple[str, str], tuple[float, float, int]] = {}
    for dataset, items in selected.items():
        batches = range(0, len(items), settings.batch_size)
        for start in tqdm(batches, desc=f"Llama PPL {dataset}"):
            batch = items[start : start + settings.batch_size]
            batch_scores = compute_conditional_llama_ppl_batch(
                model,
                tokenizer,
                batch,
                device=device,
                use_chat_template=use_chat,
            )
            for item, score in zip(batch, batch_scores, strict=True):
                scores[(dataset, item.item_id)] = score
    _release_model(model, device)
    del model
    return scores


def _score_one_item(  # noqa: PLR0913
    *,
    config: Config,
    settings: ExperimentSettings,
    scorer: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    item: ReferenceItem,
    mask_ratio: float,
    mask_token_id: int,
    device: torch.device,
) -> tuple[list[float], list[float], int, int]:
    use_chat = model_name == "dream" or (model_name == "llada" and "instruct" in config.llada_model_path.lower())
    ids, prompt_length, completion_length = tokenize_prompt_completion(
        tokenizer,
        item.prompt_text,
        item.completion_text,
        use_chat_template=use_chat,
    )
    masked_ids, completion_masks, num_masked = build_mask_draws(
        ids,
        prompt_length=prompt_length,
        mask_token_id=mask_token_id,
        mask_ratio=mask_ratio,
        num_draws=settings.mask_draws,
        seed_parts=(config.seed, item.dataset, item.item_id, model_name, mask_ratio),
    )

    entropy_draws: list[float] = []
    self_draws: list[float] = []
    for start in range(0, settings.mask_draws, settings.batch_size):
        batch_ids = masked_ids[start : start + settings.batch_size].to(device)
        batch_masks = completion_masks[start : start + settings.batch_size].to(device)
        if model_name == "llada":
            entropy, self_certainty = score_llada_mask_batch(
                cast(LLaDAModelLM, scorer),
                batch_ids,
                batch_masks,
                prompt_length=prompt_length,
            )
        else:
            entropy, self_certainty = score_dream_mask_batch(
                cast(DreamModel, scorer),
                batch_ids,
                batch_masks,
                completion_length=completion_length,
            )
        entropy_draws.extend(entropy.detach().cpu().tolist())
        self_draws.extend(self_certainty.detach().cpu().tolist())
    return entropy_draws, self_draws, completion_length, num_masked


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    settings, remaining = parse_experiment_settings(sys.argv[1:])
    config = config_from_remaining_args(remaining)
    configure_runtime(config)
    models = _parse_names(
        settings.models,
        supported=SUPPORTED_MODELS,
        field_name="score_correlation_models",
    )
    datasets = _parse_names(
        settings.datasets,
        supported=SUPPORTED_DATASETS,
        field_name="score_correlation_datasets",
    )
    ratios = parse_mask_ratios(settings.mask_ratios)

    print("Partial-hypothesis score experiment")
    print("  entropy certainty = KL(p || uniform) / log(V)")
    print("  self-certainty    = KL(uniform || p) / log(V)")
    print("  higher proxy scores are better; lower Llama PPL is better")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer_tokenizers = {
        "llada": cast(PreTrainedTokenizerBase, get_tokenizer(config, "llada")),
        "dream": cast(PreTrainedTokenizerBase, get_tokenizer(config, "dream")),
    }

    selected: dict[str, list[ReferenceItem]] = {}
    eligibility: dict[str, EligibilityCounts] = {}
    for dataset in datasets:
        source = load_reference_items(config, dataset)
        selected[dataset], eligibility[dataset] = select_shared_eligible_items(
            source,
            tokenizers=scorer_tokenizers,
            config=config,
            num_items=settings.num_items,
            min_completion_tokens=settings.min_completion_tokens,
        )
        counts = eligibility[dataset]
        print(
            f"{dataset}: selected {len(selected[dataset])} shared items "
            f"from {counts.eligible_items}/{counts.source_items} eligible/source rows",
        )

    signature = _experiment_signature(
        config,
        settings=settings,
        models=models,
        datasets=datasets,
        ratios=ratios,
        selected=selected,
    )
    prefix = settings.output_prefix
    points_path = Path(config.results_dir) / f"{prefix}_points.csv"
    correlations_path = Path(config.results_dir) / f"{prefix}_correlations.csv"
    existing = load_resumable_points(points_path, signature)
    point_rows = existing.to_dict("records") if not existing.empty else []
    completed_keys = {
        (str(row["model"]), str(row["dataset"]), str(row["item_id"]), float(row["mask_ratio"]))
        for row in point_rows
    }
    if completed_keys:
        print(f"Resuming with {len(completed_keys)} completed correlation points from {points_path}")

    expected_keys = {
        (model_name, dataset, item.item_id, float(mask_ratio))
        for model_name in models
        for dataset in datasets
        for mask_ratio in ratios
        for item in selected[dataset]
    }
    remaining_keys = expected_keys - completed_keys
    llama_scores = _llama_scores_for_items(config, settings, selected, device=device) if remaining_keys else {}
    rows_since_flush = 0
    for model_name in models:
        if not any(key[0] == model_name for key in remaining_keys):
            print(f"Skipping fully completed {model_name} scorer")
            continue
        print(f"Loading {model_name} scorer")
        scorer = _load_scorer(config, model_name, device)
        tokenizer = scorer_tokenizers[model_name]
        mask_token_id = getattr(getattr(scorer, "config", None), "mask_token_id", None)
        if not isinstance(mask_token_id, int):
            raise ValueError(f"{model_name} checkpoint does not expose an integer mask_token_id")

        for dataset in datasets:
            counts = eligibility[dataset]
            for mask_ratio in ratios:
                description = f"{model_name} {dataset} mask={mask_ratio:.2f}"
                for item in tqdm(selected[dataset], desc=description):
                    key = (model_name, dataset, item.item_id, float(mask_ratio))
                    if key in completed_keys:
                        continue
                    entropy_draws, self_draws, completion_tokens, num_masked = _score_one_item(
                        config=config,
                        settings=settings,
                        scorer=scorer,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        item=item,
                        mask_ratio=mask_ratio,
                        mask_token_id=mask_token_id,
                        device=device,
                    )
                    llama_nll, llama_ppl, llama_tokens = llama_scores[(dataset, item.item_id)]
                    row: dict[str, Any] = {
                        "experiment_signature": signature,
                        "model": model_name,
                        "dataset": dataset,
                        "task_family": item.task_family,
                        "item_id": item.item_id,
                        "dataset_index": item.dataset_index,
                        "mask_ratio": mask_ratio,
                        "realized_mask_ratio": num_masked / completion_tokens,
                        "mask_draws": settings.mask_draws,
                        "completion_tokens": completion_tokens,
                        "masked_tokens_per_draw": num_masked,
                        "llama_completion_tokens": llama_tokens,
                        "llama_mean_nll": llama_nll,
                        "llama_ppl": llama_ppl,
                        "entropy_certainty_mean": float(np.mean(entropy_draws)),
                        "entropy_certainty_sd": float(np.std(entropy_draws, ddof=1)),
                        "self_certainty_mean": float(np.mean(self_draws)),
                        "self_certainty_sd": float(np.std(self_draws, ddof=1)),
                        "source_items": counts.source_items,
                        "eligible_items": counts.eligible_items,
                        "prompt_text": item.prompt_text,
                        "completion_text": item.completion_text,
                    }
                    for draw_index, value in enumerate(entropy_draws):
                        row[f"entropy_draw_{draw_index:02d}"] = value
                    for draw_index, value in enumerate(self_draws):
                        row[f"self_certainty_draw_{draw_index:02d}"] = value
                    point_rows.append(row)
                    completed_keys.add(key)
                    remaining_keys.discard(key)
                    rows_since_flush += 1
                    if rows_since_flush >= settings.batch_size:
                        _atomic_write_csv(pd.DataFrame(point_rows), points_path)
                        rows_since_flush = 0

        _release_model(scorer, device)
        del scorer

    points = pd.DataFrame(point_rows)
    if completed_keys != expected_keys:
        missing = sorted(expected_keys - completed_keys)[:5]
        unexpected = sorted(completed_keys - expected_keys)[:5]
        raise RuntimeError(
            f"Point coverage mismatch: expected={len(expected_keys)}, completed={len(completed_keys)}, "
            f"missing examples={missing}, unexpected examples={unexpected}",
        )
    _atomic_write_csv(points, points_path)
    correlations = summarize_correlations(
        points,
        bootstrap_samples=settings.bootstrap_samples,
        seed=config.seed,
    )
    _atomic_write_csv(correlations, correlations_path)
    print(f"Saved {len(points)} independent points to {points_path}")
    print(f"Saved {len(correlations)} condition summaries to {correlations_path}")


if __name__ == "__main__":
    main()

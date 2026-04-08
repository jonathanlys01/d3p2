"""
CFG collapse experiment on TruthfulQA for LLaDA.

For each CFG value in ``linspace(1, 3, 9)``:
1. Baseline: generate 3 answers independently.
2. Our method: run real D5P4 sampling with 9 live paths arranged as 3 groups of 3
   under transversal subset selection during denoising.
3. At the end, extract 3 answers by taking one representative per final group.
   If a group has not fully collapsed to one unique string, fall back to the
   highest-scoring member inside that group.

The output JSON stores per-CFG metrics for:
- the raw 9-path D5P4 pool
- the extracted 3-answer D5P4 subset
- the independent 3-sample baseline
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import idr_torch
from d5p4.config import RESULTS_DIR, Cache, Config
from d5p4.data.qa import get_qa_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import Evaluator
from d5p4.subsample import get_subsample_selector
from d5p4.utils import compile_model, seed_all
from d5p4.utils import print as u_print


DEFAULT_SELECTOR_METHOD = "greedy_map"
SWEEP_CFG_VALUES = np.linspace(1.0, 3.0, num=9).tolist()
SUMMARY_KEYS = ("f1", "bleu", "cos_at_k", "distinct_2", "self_bleu", "wasserstein_distance")


def _build_config(base_cfg: Config, **updates) -> Config:
    cfg_dict = asdict(base_cfg)
    cfg_dict.update(updates)
    cfg_dict["disable_sys_args"] = True
    return Config(**cfg_dict)


def _decode_completions(sampler: LLADASampler, prompt: str, sample_ids: torch.Tensor) -> tuple[list[str], int]:
    prompt_tokens = sampler._preprocess_prompt(prompt)
    prompt_len = prompt_tokens.shape[1]

    decoded: list[str] = []
    for sample in sample_ids:
        completion_tokens = sample[prompt_len:]
        text = sampler.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
        decoded.append(text)

    return decoded, prompt_len


def _build_answer_cache(sampler: LLADASampler, prompt: str, sample_ids: torch.Tensor) -> Cache:
    prompt_tokens = sampler._preprocess_prompt(prompt)
    prompt_len = prompt_tokens.shape[1]
    sample_ids = sample_ids.to(sampler.device)

    with torch.no_grad():
        logits, all_hidden = sampler._forward_model(sample_ids)
        embeddings = all_hidden[-1]
        log_p_x0 = F.log_softmax(logits, dim=-1)

    return Cache(
        log_p_x0=log_p_x0[:, prompt_len:],
        embeddings=embeddings[:, prompt_len:],
        x=sample_ids[:, prompt_len:],
    )


def _extract_group_representatives(
    sampler: LLADASampler,
    prompt: str,
    sample_ids: torch.Tensor,
    selector_cfg: Config,
) -> tuple[list[str], list[str], list[int], list[dict]]:
    decoded, _ = _decode_completions(sampler, prompt, sample_ids)
    selector = get_subsample_selector(selector_cfg)
    cache = _build_answer_cache(sampler, prompt, sample_ids)
    scores = selector.compute_scores(cache)
    if scores is None:
        raise RuntimeError("Could not compute final scores for grouped representative extraction.")

    group_size = selector_cfg.group_size
    n_groups = selector_cfg.n_groups
    score_list = scores.detach().cpu().tolist()

    selected_texts: list[str] = []
    selected_indices: list[int] = []
    group_records: list[dict] = []

    for group_idx in range(n_groups):
        start = group_idx * group_size
        end = start + group_size
        group_texts = decoded[start:end]
        group_scores = score_list[start:end]
        unique_texts = list(dict.fromkeys(group_texts))
        fully_collapsed = len(unique_texts) == 1

        if fully_collapsed:
            local_idx = 0
            selection_reason = "collapsed"
        else:
            local_idx = int(np.argmax(group_scores))
            selection_reason = "group_argmax"

        global_idx = start + local_idx
        selected_indices.append(global_idx)
        selected_texts.append(decoded[global_idx])
        group_records.append(
            {
                "group_index": group_idx,
                "group_indices": list(range(start, end)),
                "texts": group_texts,
                "scores": group_scores,
                "unique_count": len(unique_texts),
                "fully_collapsed": fully_collapsed,
                "selection_reason": selection_reason,
                "selected_index": global_idx,
            },
        )

    return decoded, selected_texts, selected_indices, group_records


def _evaluate_bundle(
    evaluator: Evaluator,
    generations: list[list[str]],
    good_refs: list[list[str]],
    bad_refs: list[list[str]],
) -> dict:
    metrics = evaluator.evaluate(generations, references=good_refs)
    wd_good_scores: list[float] = []
    wd_bad_scores: list[float] = []
    for group_gen, group_good, group_bad in zip(generations, good_refs, bad_refs):
        if not group_gen or not group_good:
            continue
        wd_good, wd_bad = evaluator.compute_wasserstein_distance(group_gen, group_good, group_bad)
        wd_good_scores.append(float(wd_good))
        if not np.isnan(wd_bad):
            wd_bad_scores.append(float(wd_bad))

    if wd_good_scores:
        metrics["avg_wd_good"] = sum(wd_good_scores) / len(wd_good_scores)
    if wd_bad_scores:
        metrics["avg_wd_bad"] = sum(wd_bad_scores) / len(wd_bad_scores)
    return metrics


def _save_results(output_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else x)


def run_cfg_collapse_experiment(cfg: Config, cfg_values: list[float] | None = None) -> tuple[dict, str]:  # noqa: C901, PLR0915
    if cfg.model != "llada":
        raise ValueError(f"This experiment only supports model='llada', got {cfg.model!r}")
    if cfg.qa_dataset != "truthful_qa":
        raise ValueError(f"This experiment is restricted to qa_dataset='truthful_qa', got {cfg.qa_dataset!r}")

    cfg_values = cfg_values or SWEEP_CFG_VALUES
    selector_method = cfg.method.lower()
    if selector_method in {"baseline", "random"}:
        selector_method = DEFAULT_SELECTOR_METHOD
    selector_cfg = _build_config(
        cfg,
        method=selector_method,
        model="llada",
        n_groups=3,
        group_size=3,
        transversal=True,
    )

    sampler_cfg = _build_config(
        cfg,
        method=selector_method,
        model="llada",
        n_groups=3,
        group_size=3,
        transversal=True,
    )
    sampler = LLADASampler(sampler_cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"cfg_collapse_truthfulqa_{timestamp}.json")

    results: dict = {
        "config": asdict(cfg),
        "selector_method": selector_method,
        "selector_config": asdict(selector_cfg),
        "cfg_values": cfg_values,
        "n_prompts": len(dataset),
        "results_by_cfg": {},
    }

    u_print(f"Running CFG collapse sweep over {len(cfg_values)} values on {len(dataset)} TruthfulQA prompts")
    u_print(f"D5P4 selector: {selector_method} with transversal=True (3 groups x 3 live paths)")
    u_print(f"Saving incremental results to {output_path}")

    try:
        for cfg_idx, cfg_value in enumerate(cfg_values):
            u_print(f"\n{'=' * 72}")
            u_print(f"[{cfg_idx + 1}/{len(cfg_values)}] CFG scale = {cfg_value:.2f}")
            u_print(f"{'=' * 72}")

            d5p4_cfg = _build_config(
                cfg,
                method=selector_method,
                model="llada",
                cfg_scale=cfg_value,
                n_groups=3,
                group_size=3,
                transversal=True,
            )
            baseline_cfg = _build_config(
                cfg,
                method="baseline",
                model="llada",
                cfg_scale=cfg_value,
                n_groups=3,
                group_size=1,
                transversal=False,
            )

            pooled_generations: list[list[str]] = []
            selected_generations: list[list[str]] = []
            baseline_generations: list[list[str]] = []
            good_refs: list[list[str]] = []
            bad_refs: list[list[str]] = []
            sample_records: list[dict] = []

            for prompt_idx, row in enumerate(dataset.itertuples()):
                prompt: str = row.question  # type: ignore[assignment]
                correct_answers: list[str] = row.correct_answers  # type: ignore[assignment]
                incorrect_answers: list[str] = row.incorrect_answers  # type: ignore[assignment]

                u_print(
                    f"[cfg {cfg_idx + 1}/{len(cfg_values)} | prompt {prompt_idx + 1}/{len(dataset)}] {prompt[:80]}...",
                    verbose=True,
                )

                sampler.update_config(d5p4_cfg)
                seed_all(int(cfg.seed + cfg_idx * 100_000 + prompt_idx * 100))
                d5p4_ids = sampler.sample(prompt=prompt)
                d5p4_texts, selected_texts, selected_indices, group_records = _extract_group_representatives(
                    sampler,
                    prompt,
                    d5p4_ids,
                    selector_cfg,
                )

                sampler.update_config(baseline_cfg)
                seed_all(int(cfg.seed + cfg_idx * 100_000 + prompt_idx * 100 + 1))
                baseline_ids = sampler.sample(prompt=prompt)
                baseline_texts, _ = _decode_completions(sampler, prompt, baseline_ids)

                pooled_generations.append(d5p4_texts)
                selected_generations.append(selected_texts)
                baseline_generations.append(baseline_texts)
                good_refs.append(correct_answers)
                bad_refs.append(incorrect_answers)

                sample_records.append(
                    {
                        "question": prompt,
                        "correct_answers": correct_answers,
                        "incorrect_answers": incorrect_answers,
                        "d5p4_paths_9": d5p4_texts,
                        "selected_indices": selected_indices,
                        "d5p4_selected_3": selected_texts,
                        "group_records": group_records,
                        "baseline_3": baseline_texts,
                    },
                )

            pool_metrics = _evaluate_bundle(evaluator, pooled_generations, good_refs, bad_refs)
            selected_metrics = _evaluate_bundle(evaluator, selected_generations, good_refs, bad_refs)
            baseline_metrics = _evaluate_bundle(evaluator, baseline_generations, good_refs, bad_refs)
            total_groups = sum(len(record["group_records"]) for record in sample_records)
            collapsed_groups = sum(
                int(group["fully_collapsed"]) for record in sample_records for group in record["group_records"]
            )
            fallback_groups = sum(
                int(group["selection_reason"] == "group_argmax")
                for record in sample_records
                for group in record["group_records"]
            )
            collapse_summary = {
                "collapsed_groups": float(collapsed_groups),
                "collapse_rate": collapsed_groups / total_groups if total_groups else 0.0,
                "fallback_groups": float(fallback_groups),
                "fallback_rate": fallback_groups / total_groups if total_groups else 0.0,
            }
            deltas = {}
            for key in SUMMARY_KEYS:
                left = selected_metrics.get(key)
                right = baseline_metrics.get(key)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    deltas[key] = float(left - right)

            cfg_result = {
                "d5p4_pool9_metrics": pool_metrics,
                "d5p4_selected3_metrics": selected_metrics,
                "baseline3_metrics": baseline_metrics,
                "d5p4_minus_baseline": deltas,
                "collapse_summary": collapse_summary,
                "samples": sample_records,
            }
            results["results_by_cfg"][str(cfg_value)] = cfg_result
            _save_results(output_path, results)
            u_print(f"\nCFG={cfg_value:.2f}")
            for key in SUMMARY_KEYS:
                selected = selected_metrics.get(key)
                baseline = baseline_metrics.get(key)
                if isinstance(selected, (int, float)) and isinstance(baseline, (int, float)):
                    u_print(
                        f"  {key:22} selected={selected:8.4f} baseline={baseline:8.4f} "
                        f"delta={selected - baseline:+8.4f}",
                    )

    finally:
        if sampler.distributed_utils:
            sampler.distributed_utils.cleanup()

    return results, output_path


def main() -> None:
    cfg = Config()
    results, output_path = run_cfg_collapse_experiment(cfg)

    if idr_torch.rank == 0:
        _save_results(output_path, results)
        u_print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

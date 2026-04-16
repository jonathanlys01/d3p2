"""
CFG sweep on TruthfulQA comparing independent-N selection against grouped greedy_map.

For each CFG value:
1. Generate N = k * m independent answers with baseline sampling.
2. Select k answers from the same independent pool via:
   - k best of N by F1
   - k best of N by PPL
   - random k-subset (k indep)
3. Run grouped greedy_map decoding with k groups of size m for several interaction
   weights, then collapse each group to one final representative. When a group has
   not fully collapsed, the fallback representative is chosen with
   ``eval_selection_metric``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from d5p4.config import Cache, Config
from d5p4.data.qa import get_qa_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import Evaluator
from d5p4.subsample import get_subsample_selector
from d5p4.utils import compile_model, is_primary_process, seed_all
from d5p4.utils import print as u_print


DEFAULT_SELECTOR_METHOD = "greedy_map"
DEFAULT_CFG_VALUES = np.linspace(1.0, 3.0, num=9).tolist()
DEFAULT_GREEDY_MAP_INTERACTIONS = [0.0, 5.0, 20.0]
INDEPENDENT_SELECTIONS = (
    ("k_best_of_n_f1", "k best of N (F1)", "f1"),
    ("k_best_of_n_ppl", "k best of N (PPL)", "ppl"),
    ("k_indep", "k indep", "random"),
)
SUMMARY_KEYS = ("f1", "perplexity", "distinct_2", "self_bleu", "cos_at_k", "avg_wd_good")


def _build_config(base_cfg: Config, **updates) -> Config:
    cfg_dict = asdict(base_cfg)
    cfg_dict.update(updates)
    cfg_dict["disable_sys_args"] = True
    return Config(**cfg_dict)


def _parse_float_list_env(env_name: str, default: list[float]) -> list[float]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default

    values = [token.strip() for token in raw.replace(";", ",").split(",")]
    parsed = [float(value) for value in values if value]
    return parsed or default


def _get_results_output_dir(cfg: Config) -> str:
    subdir = os.getenv("CFG_COLLAPSE_RESULTS_SUBDIR", "cfg_collapse_truthfulqa").strip().strip("/")
    return os.path.join(cfg.results_dir, subdir) if subdir else cfg.results_dir


def _sanitize_path_component(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return sanitized.strip("._") or "item"


def _strategy_key_for_interaction(interaction: float) -> str:
    label = f"{interaction:g}".replace("-", "m").replace(".", "p")
    return f"greedy_map_w{label}"


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


def _select_top_k_indices(scores: list[float], k: int, reverse_sort: bool) -> list[int]:
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=reverse_sort)
    return sorted(idx for idx, _ in ranked[:k])


def _collapse_summary(counts: dict[str, int]) -> dict[str, float]:
    total_groups = counts["total_groups"]
    return {
        "collapsed_groups": float(counts["collapsed_groups"]),
        "collapse_rate": counts["collapsed_groups"] / total_groups if total_groups else 0.0,
        "fallback_groups": float(counts["fallback_groups"]),
        "fallback_rate": counts["fallback_groups"] / total_groups if total_groups else 0.0,
    }


def _extract_group_representatives(  # noqa: PLR0913
    evaluator: Evaluator,
    sampler: LLADASampler,
    prompt: str,
    sample_ids: torch.Tensor,
    selector_cfg: Config,
    references: list[str],
) -> tuple[list[str], list[str], list[int], list[dict]]:
    decoded, _ = _decode_completions(sampler, prompt, sample_ids)
    selection_metric = selector_cfg.eval_selection_metric.lower()

    internal_scores = None
    reference_groups = [references] if selection_metric == "f1" else None
    if selection_metric == "int":
        selector = get_subsample_selector(selector_cfg)
        scores = selector.compute_scores(_build_answer_cache(sampler, prompt, sample_ids))
        if scores is None:
            raise RuntimeError("Could not compute internal scores for grouped representative extraction.")
        internal_scores = [scores.detach().cpu().tolist()]

    score_groups, reverse_sort = evaluator.score_baseline_candidates(
        [decoded],
        metric=selection_metric,
        references=reference_groups,
        internal_scores=internal_scores,
    )
    score_list = score_groups[0]

    selected_texts: list[str] = []
    selected_indices: list[int] = []
    group_records: list[dict] = []

    for group_idx in range(selector_cfg.n_groups):
        start = group_idx * selector_cfg.group_size
        end = start + selector_cfg.group_size
        group_texts = decoded[start:end]
        group_scores = score_list[start:end]
        unique_texts = list(dict.fromkeys(group_texts))
        fully_collapsed = len(unique_texts) == 1

        if fully_collapsed:
            local_idx = 0
            selection_reason = "collapsed"
        else:
            local_idx = _select_top_k_indices(group_scores, 1, reverse_sort)[0]
            selection_reason = f"{selection_metric}_fallback"

        global_idx = start + local_idx
        selected_indices.append(global_idx)
        selected_texts.append(decoded[global_idx])
        group_records.append(
            {
                "group_index": group_idx,
                "group_indices": list(range(start, end)),
                "texts": group_texts,
                "scores": group_scores,
                "score_metric": selection_metric,
                "unique_count": len(unique_texts),
                "fully_collapsed": fully_collapsed,
                "selection_reason": selection_reason,
                "selected_index": global_idx,
            },
        )

    return decoded, selected_texts, selected_indices, group_records


def _select_independent_candidates(
    evaluator: Evaluator,
    candidates: list[str],
    references: list[str],
    subset_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    selected_by_strategy: dict[str, list[str]] = {}
    selection_records: dict[str, dict] = {}

    for strategy_key, _display_name, metric in INDEPENDENT_SELECTIONS:
        score_groups, reverse_sort = evaluator.score_baseline_candidates(
            [candidates],
            metric=metric,
            references=[references] if metric == "f1" else None,
            random_seed=random_seed if metric == "random" else None,
        )
        score_list = score_groups[0]
        selected_indices = _select_top_k_indices(score_list, subset_size, reverse_sort)
        selected_texts = [candidates[idx] for idx in selected_indices]

        selected_by_strategy[strategy_key] = selected_texts
        selection_records[strategy_key] = {
            "selection_metric": metric,
            "scores": score_list,
            "selected_indices": selected_indices,
            "selected_texts": selected_texts,
        }

    return selected_by_strategy, selection_records


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


def _strategy_export_payload(  # noqa: PLR0913
    *,
    export_cfg: Config,
    generation_cfg: Config,
    strategy_key: str,
    strategy_label: str,
    cfg_value: float,
    selected_texts: list[list[str]],
    raw_texts: list[list[str]],
    metrics: dict,
    references: list[list[str]],
    selection_metadata: dict,
    source_aggregate_path: str,
) -> dict:
    return {
        "config": asdict(export_cfg),
        "generation_config": asdict(generation_cfg),
        "metrics": metrics,
        "text_samples": selected_texts,
        "eval_text_samples": selected_texts,
        "raw_text_samples": raw_texts,
        "references": references,
        "cfg_scale": cfg_value,
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "selection_metadata": selection_metadata,
        "source_aggregate_file": os.path.basename(source_aggregate_path),
    }


def _export_strategy_results(  # noqa: PLR0913
    export_dir: str,
    aggregate_output_path: str,
    cfg_value: float,
    sample_records: list[dict],
    good_refs: list[list[str]],
    strategy_results: dict[str, dict],
    independent_cfg: Config,
    greedy_cfgs: dict[str, Config],
    greedy_labels: dict[str, str],
) -> dict[str, str]:
    os.makedirs(export_dir, exist_ok=True)
    export_paths: dict[str, str] = {}

    independent_raw = [record["independent_pool_n"] for record in sample_records]
    independent_selected = {
        strategy_key: [record["independent_selected"][strategy_key]["selected_texts"] for record in sample_records]
        for strategy_key, _label, _metric in INDEPENDENT_SELECTIONS
    }

    for strategy_key, strategy_label, metric in INDEPENDENT_SELECTIONS:
        selected_count = len(independent_selected[strategy_key][0]) if independent_selected[strategy_key] else 0
        export_cfg = _build_config(
            independent_cfg,
            method="baseline",
            n_groups=selected_count,
            group_size=1,
            transversal=False,
            eval_selection_metric=metric,
        )
        payload = _strategy_export_payload(
            export_cfg=export_cfg,
            generation_cfg=independent_cfg,
            strategy_key=strategy_key,
            strategy_label=strategy_label,
            cfg_value=cfg_value,
            selected_texts=independent_selected[strategy_key],
            raw_texts=independent_raw,
            metrics=strategy_results[strategy_key]["metrics"],
            references=good_refs,
            selection_metadata={
                "selection_type": "independent_posthoc",
                "selection_metric": metric,
                "candidate_count": independent_cfg.n_groups,
                "selected_count": export_cfg.n_groups,
            },
            source_aggregate_path=aggregate_output_path,
        )
        file_name = f"cfg_{_sanitize_path_component(f'{cfg_value:g}')}_{strategy_key}.json"
        path = os.path.join(export_dir, file_name)
        _save_results(path, payload)
        export_paths[strategy_key] = path

    for strategy_key, greedy_cfg in greedy_cfgs.items():
        strategy_label = greedy_labels[strategy_key]
        raw_texts = [record["greedy_map"][strategy_key]["pool_texts"] for record in sample_records]
        selected_texts = [record["greedy_map"][strategy_key]["selected_texts"] for record in sample_records]
        export_cfg = _build_config(
            greedy_cfg,
            n_groups=greedy_cfg.n_groups,
            group_size=1,
            transversal=False,
        )
        payload = _strategy_export_payload(
            export_cfg=export_cfg,
            generation_cfg=greedy_cfg,
            strategy_key=strategy_key,
            strategy_label=strategy_label,
            cfg_value=cfg_value,
            selected_texts=selected_texts,
            raw_texts=raw_texts,
            metrics=strategy_results[strategy_key]["metrics"],
            references=good_refs,
            selection_metadata={
                "selection_type": "group_representatives",
                "representative_metric": greedy_cfg.eval_selection_metric,
                "interaction": greedy_cfg._w_interaction,
                "candidate_count": greedy_cfg.n_groups * greedy_cfg.group_size,
                "selected_count": export_cfg.n_groups,
            },
            source_aggregate_path=aggregate_output_path,
        )
        file_name = f"cfg_{_sanitize_path_component(f'{cfg_value:g}')}_{strategy_key}.json"
        path = os.path.join(export_dir, file_name)
        _save_results(path, payload)
        export_paths[strategy_key] = path

    return export_paths


def run_cfg_collapse_experiment(cfg: Config, cfg_values: list[float] | None = None) -> tuple[dict, str]:  # noqa: C901, PLR0912, PLR0915
    if cfg.model != "llada":
        raise ValueError(f"This experiment only supports model='llada', got {cfg.model!r}")
    if cfg.qa_dataset != "truthful_qa":
        raise ValueError(f"This experiment is restricted to qa_dataset='truthful_qa', got {cfg.qa_dataset!r}")
    if cfg.group_size <= 1:
        raise ValueError(f"This experiment expects group_size > 1, got {cfg.group_size}")

    cfg_values = cfg_values or _parse_float_list_env("CFG_VALUES", DEFAULT_CFG_VALUES)
    interaction_values = _parse_float_list_env("GREEDY_MAP_INTERACTIONS", DEFAULT_GREEDY_MAP_INTERACTIONS)
    selector_method = cfg.method.lower()
    if selector_method in {"baseline", "random"}:
        selector_method = DEFAULT_SELECTOR_METHOD

    subset_size = cfg.n_groups
    group_size = cfg.group_size
    total_candidates = subset_size * group_size
    greedy_strategy_defs = [
        (_strategy_key_for_interaction(interaction), f"greedy_map (w={interaction:g})", interaction)
        for interaction in interaction_values
    ]
    greedy_strategy_labels = {strategy_key: label for strategy_key, label, _interaction in greedy_strategy_defs}
    strategy_order = [strategy_key for strategy_key, _, _ in INDEPENDENT_SELECTIONS] + [
        strategy_key for strategy_key, _, _ in greedy_strategy_defs
    ]

    sampler_cfg = _build_config(
        cfg,
        method=selector_method,
        model="llada",
        n_groups=subset_size,
        group_size=group_size,
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
    output_dir = _get_results_output_dir(cfg)
    standard_export_dir = os.path.join(output_dir, "standard_exports")
    cfg_tag = f"{cfg_values[0]:g}-{cfg_values[-1]:g}" if cfg_values else "empty"
    output_path = os.path.join(output_dir, f"cfg_collapse_truthfulqa_{cfg_tag}_{timestamp}.json")

    results: dict = {
        "config": asdict(cfg),
        "cfg_values": cfg_values,
        "n_prompts": len(dataset),
        "output_dir": output_dir,
        "standard_export_dir": standard_export_dir,
        "comparison_setup": {
            "k": subset_size,
            "n": total_candidates,
            "group_size": group_size,
            "independent_selection_strategies": [
                {"key": strategy_key, "label": label, "metric": metric}
                for strategy_key, label, metric in INDEPENDENT_SELECTIONS
            ],
            "greedy_map_method": selector_method,
            "greedy_map_interactions": interaction_values,
            "greedy_map_representative_metric": cfg.eval_selection_metric,
        },
        "results_by_cfg": {},
    }

    u_print(f"Running CFG sweep over {len(cfg_values)} values on {len(dataset)} TruthfulQA prompts")
    u_print(
        f"Comparing one independent N={total_candidates} pool against greedy_map "
        f"({subset_size} groups x {group_size}, representative metric={cfg.eval_selection_metric})",
    )
    u_print(f"Saving incremental results to {output_path}")

    try:
        for cfg_idx, cfg_value in enumerate(cfg_values):
            u_print(f"\n{'=' * 72}")
            u_print(f"[{cfg_idx + 1}/{len(cfg_values)}] CFG scale = {cfg_value:.2f}")
            u_print(f"{'=' * 72}")

            independent_cfg = _build_config(
                cfg,
                method="baseline",
                model="llada",
                cfg_scale=cfg_value,
                n_groups=total_candidates,
                group_size=1,
                transversal=False,
            )
            greedy_cfgs = {
                strategy_key: _build_config(
                    cfg,
                    method=selector_method,
                    model="llada",
                    cfg_scale=cfg_value,
                    n_groups=subset_size,
                    group_size=group_size,
                    transversal=True,
                    _w_interaction=interaction,
                )
                for strategy_key, _label, interaction in greedy_strategy_defs
            }

            strategy_generations: dict[str, list[list[str]]] = {strategy_key: [] for strategy_key in strategy_order}
            collapse_counts = {
                strategy_key: {"total_groups": 0, "collapsed_groups": 0, "fallback_groups": 0}
                for strategy_key, _, _ in greedy_strategy_defs
            }
            good_refs: list[list[str]] = []
            bad_refs: list[list[str]] = []
            sample_records: list[dict] = []

            for prompt_idx, row in enumerate(dataset.itertuples()):
                prompt: str = row.question  # type: ignore[assignment]
                correct_answers: list[str] = row.correct_answers  # type: ignore[assignment]
                incorrect_answers: list[str] = row.incorrect_answers  # type: ignore[assignment]
                cfg_seed_key = int(round(cfg_value * 1_000))
                prompt_seed = int(cfg.seed + cfg_seed_key * 100_000 + prompt_idx * 1_000)

                u_print(
                    f"[cfg {cfg_idx + 1}/{len(cfg_values)} | prompt {prompt_idx + 1}/{len(dataset)}] {prompt[:80]}...",
                    verbose=True,
                )

                sampler.update_config(independent_cfg)
                seed_all(prompt_seed)
                independent_ids = sampler.sample(prompt=prompt)
                independent_texts, _ = _decode_completions(sampler, prompt, independent_ids)
                independent_selected, independent_selection_records = _select_independent_candidates(
                    evaluator,
                    independent_texts,
                    correct_answers,
                    subset_size,
                    random_seed=prompt_seed + 17,
                )

                for strategy_key, selected_texts in independent_selected.items():
                    strategy_generations[strategy_key].append(selected_texts)

                greedy_records: dict[str, dict] = {}
                for strategy_key, _label, _interaction in greedy_strategy_defs:
                    sampler.update_config(greedy_cfgs[strategy_key])
                    seed_all(prompt_seed)
                    grouped_ids = sampler.sample(prompt=prompt)
                    grouped_texts, selected_texts, selected_indices, group_records = _extract_group_representatives(
                        evaluator,
                        sampler,
                        prompt,
                        grouped_ids,
                        greedy_cfgs[strategy_key],
                        correct_answers,
                    )

                    strategy_generations[strategy_key].append(selected_texts)
                    collapse_counts[strategy_key]["total_groups"] += len(group_records)
                    collapse_counts[strategy_key]["collapsed_groups"] += sum(
                        int(group["fully_collapsed"]) for group in group_records
                    )
                    collapse_counts[strategy_key]["fallback_groups"] += sum(
                        int(group["selection_reason"] != "collapsed") for group in group_records
                    )
                    greedy_records[strategy_key] = {
                        "interaction": greedy_cfgs[strategy_key]._w_interaction,
                        "pool_texts": grouped_texts,
                        "selected_indices": selected_indices,
                        "selected_texts": selected_texts,
                        "group_records": group_records,
                    }

                good_refs.append(correct_answers)
                bad_refs.append(incorrect_answers)
                sample_records.append(
                    {
                        "question": prompt,
                        "correct_answers": correct_answers,
                        "incorrect_answers": incorrect_answers,
                        "independent_pool_n": independent_texts,
                        "independent_selected": independent_selection_records,
                        "greedy_map": greedy_records,
                    },
                )

            strategy_results = {}
            for strategy_key in strategy_order:
                metrics = _evaluate_bundle(evaluator, strategy_generations[strategy_key], good_refs, bad_refs)
                strategy_result = {"metrics": metrics}
                if strategy_key in collapse_counts:
                    strategy_result["collapse_summary"] = _collapse_summary(collapse_counts[strategy_key])
                strategy_results[strategy_key] = strategy_result

            cfg_result = {
                "strategies": strategy_results,
                "samples": sample_records,
            }
            export_paths = _export_strategy_results(
                standard_export_dir,
                output_path,
                cfg_value,
                sample_records,
                good_refs,
                strategy_results,
                independent_cfg,
                greedy_cfgs,
                greedy_strategy_labels,
            )
            cfg_result["standard_exports"] = export_paths
            results["results_by_cfg"][str(cfg_value)] = cfg_result
            _save_results(output_path, results)

            u_print(f"\nCFG={cfg_value:.2f}")
            for strategy_key, display_name, _metric in INDEPENDENT_SELECTIONS:
                metrics = strategy_results[strategy_key]["metrics"]
                summary = " ".join(
                    f"{key}={metrics[key]:.4f}" for key in SUMMARY_KEYS if isinstance(metrics.get(key), (int, float))
                )
                u_print(f"  {display_name:24} {summary}")
            for strategy_key, display_name, _interaction in greedy_strategy_defs:
                metrics = strategy_results[strategy_key]["metrics"]
                collapse = strategy_results[strategy_key]["collapse_summary"]
                summary = " ".join(
                    f"{key}={metrics[key]:.4f}" for key in SUMMARY_KEYS if isinstance(metrics.get(key), (int, float))
                )
                u_print(
                    f"  {display_name:24} {summary} "
                    f"collapse={collapse['collapse_rate']:.4f} fallback={collapse['fallback_rate']:.4f}",
                )

    finally:
        if sampler.distributed_utils:
            sampler.distributed_utils.cleanup()

    return results, output_path


def main() -> None:
    cfg = Config()
    results, output_path = run_cfg_collapse_experiment(cfg)

    if is_primary_process(cfg):
        _save_results(output_path, results)
        u_print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

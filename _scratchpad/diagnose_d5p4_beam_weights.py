#!/usr/bin/env python
"""Sweep D5P4 beam weights against exact LTR beam on the same model and device.

This is deliberately outside the pytest suite: it is a relatively expensive
diagnostic intended for occasional local or Jean Zay runs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from d5p4.config import Config
from d5p4.diffusion_llada import (
    LLADASampler,
    left_to_right_beam_sample,
    left_to_right_d5p4_beam_sample,
)
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample.greedy_map import GreedyMAPKernelSelector


DEFAULT_WEIGHTS = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    7.5e-1,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    25.0,
)


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated weight")
    if any(weight < 0.0 for weight in values):
        raise argparse.ArgumentTypeError("weights must be non-negative")
    return values


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError("seed range must be START:STOP or START:STOP:STEP")
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise argparse.ArgumentTypeError("seed range step cannot be zero")
        seeds = list(range(start, stop, step))
    else:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


def _parse_layouts(value: str) -> list[str]:
    layouts = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(layouts) - {"global", "transversal"}
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown layouts: {', '.join(sorted(invalid))}")
    if not layouts:
        raise argparse.ArgumentTypeError("expected at least one layout")
    return layouts


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _tiny_model(seed: int, device: torch.device, max_sequence_length: int) -> LLaDAModelLM:
    torch.manual_seed(seed)
    config = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=64,
        embedding_size=64,
        max_sequence_length=max_sequence_length,
        mask_token_id=63,
        rope=True,
        alibi=False,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        weight_tying=True,
        init_device="cpu",
        eos_token_id=None,
        pad_token_id=None,
    )
    return LLaDAModelLM(config, init_params=True).to(device).eval()


def _tiny_prompt(seed: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[1 + seed % 29, 30 + seed % 29]], dtype=torch.long, device=device)


def _checkpoint_prompt(seed: int) -> str:
    rng = random.Random(seed)
    boxes = rng.randint(4, 30)
    items_per_box = rng.randint(3, 20)
    removed = rng.randint(1, boxes * items_per_box // 3)
    return (
        f"A store has {boxes} boxes with {items_per_box} notebooks in each box. "
        f"It sells {removed} notebooks. How many notebooks remain? Show your reasoning."
    )


def _load_checkpoint(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, LLADASampler]:
    config = Config(
        disable_sys_args=True,
        model="llada",
        llada_model_path=args.model_path,
        llada_tokenizer=args.tokenizer or args.model_path,
        cache_dir=args.cache_dir,
        llada_decoder="classic_beam",
        method="ltr_beam",
        cfg_scale=1.0,
        logits_eos_inf=False,
        transversal=False,
        n_groups=args.beam_size,
        group_size=1,
        classic_beam_branching_factor=args.branching_factor,
        gen_length=args.generation_length,
        compile_model=False,
        standalone_job=True,
    )
    sampler = LLADASampler(config)
    if sampler.device != device.type:
        sampler.model.to(device)
        sampler.device = device.type
    model: torch.nn.Module = sampler.model
    if args.compile_model:
        # pyrefly: ignore [bad-assignment]
        model = torch.compile(model, dynamic=True)
    model.eval()
    return model, sampler


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)


def _sequence_set(sequences: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(int(token) for token in row) for row in sequences[0].detach().cpu().tolist()}


def _run_case(  # noqa: PLR0913
    *,
    model: torch.nn.Module,
    prompt: torch.LongTensor | torch.Tensor,
    attention: torch.Tensor,
    eos_token_ids: tuple[int, ...],
    generation_length: int,
    beam_size: int,
    branching_factor: int,
    num_groups: int,
    weight: float,
    baseline: tuple[torch.LongTensor, torch.FloatTensor, int],
    device: torch.device,
    autocast: bool,
) -> dict[str, Any]:
    map_calls = 0
    original_select = GreedyMAPKernelSelector.select

    def recording_select(self, kernel, selection_count):
        nonlocal map_calls
        map_calls += 1
        return original_select(self, kernel, selection_count)

    GreedyMAPKernelSelector.select = recording_select
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with _autocast_context(device, autocast):
            result = left_to_right_d5p4_beam_sample(
                model,
                prompt,
                attention,
                generation_length=generation_length,
                beam_size=beam_size,
                branching_factor=branching_factor,
                eos_token_ids=eos_token_ids,
                num_groups=num_groups,
                diversity_weight=weight,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    finally:
        GreedyMAPKernelSelector.select = original_select

    base_sequences, base_scores, base_forwards = baseline
    sequences, scores, forwards = result
    slot_matches = (sequences == base_sequences).all(dim=-1).float().mean().item()
    base_set = _sequence_set(base_sequences)
    result_set = _sequence_set(sequences)
    set_overlap = len(base_set & result_set) / max(1, len(base_set))

    if weight == 0.0:
        if map_calls > 0:
            raise AssertionError("w=0 unexpectedly called greedy MAP")
        if not torch.equal(sequences, base_sequences) or not torch.equal(scores, base_scores):
            raise AssertionError("w=0 failed exact LTR parity")
    elif map_calls == 0:
        raise AssertionError(f"w={weight:g} did not call greedy MAP")

    return {
        "weight": weight,
        "exact_tokens": torch.equal(sequences, base_sequences),
        "exact_scores": torch.equal(scores, base_scores),
        "slot_match_fraction": slot_matches,
        "candidate_set_overlap": set_overlap,
        "mean_score_delta": float((scores - base_scores).mean().item()),
        "best_score_delta": float((scores.max() - base_scores.max()).item()),
        "unique_candidates": len(result_set),
        "baseline_unique_candidates": len(base_set),
        "forward_count": forwards,
        "baseline_forward_count": base_forwards,
        "forward_count_equal": forwards == base_forwards,
        "map_calls": map_calls,
        "elapsed_s": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def _summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(case["layout"], case["weight"])].append(case)

    rows: list[dict[str, Any]] = []
    for (layout, weight), items in groups.items():
        count = len(items)
        rows.append(
            {
                "layout": layout,
                "weight": weight,
                "cases": count,
                "exact_token_fraction": sum(item["exact_tokens"] for item in items) / count,
                "exact_score_fraction": sum(item["exact_scores"] for item in items) / count,
                "mean_slot_match_fraction": sum(item["slot_match_fraction"] for item in items) / count,
                "mean_candidate_set_overlap": sum(item["candidate_set_overlap"] for item in items) / count,
                "mean_score_delta": sum(item["mean_score_delta"] for item in items) / count,
                "mean_best_score_delta": sum(item["best_score_delta"] for item in items) / count,
                "forward_count_mismatches": sum(not item["forward_count_equal"] for item in items),
                "total_map_calls": sum(item["map_calls"] for item in items),
                "total_elapsed_s": sum(item["elapsed_s"] for item in items),
                "max_peak_cuda_memory_bytes": max(
                    (item["peak_cuda_memory_bytes"] or 0 for item in items),
                    default=0,
                ),
            },
        )
    return sorted(rows, key=lambda row: (row["layout"], row["weight"]))


def _print_summaries(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'layout':12s} {'weight':>10s} {'tok=':>7s} {'score=':>7s} "
        f"{'slot':>7s} {'set':>7s} {'dmean':>11s} {'dbest':>11s} {'map':>7s} {'fwd!':>5s}",
    )
    for row in rows:
        print(
            f"{row['layout']:12s} {row['weight']:10.3g} "
            f"{row['exact_token_fraction']:7.1%} {row['exact_score_fraction']:7.1%} "
            f"{row['mean_slot_match_fraction']:7.1%} {row['mean_candidate_set_overlap']:7.1%} "
            f"{row['mean_score_delta']:11.4g} {row['mean_best_score_delta']:11.4g} "
            f"{row['total_map_calls']:7d} {row['forward_count_mismatches']:5d}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="Real checkpoint path; omit for the tiny model.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0:16"))
    parser.add_argument(
        "--weights",
        type=_parse_float_list,
        default=list(DEFAULT_WEIGHTS),
        help="Comma-separated non-negative weights.",
    )
    parser.add_argument("--layouts", type=_parse_layouts, default=["global", "transversal"])
    parser.add_argument("--generation-length", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=9)
    parser.add_argument("--branching-factor", type=int, default=9)
    parser.add_argument("--transversal-groups", type=int, default=3)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:  # noqa: C901, PLR0912
    args = _parser().parse_args()
    device = _device(args.device)
    if args.generation_length < 2:
        raise SystemExit("--generation-length must be at least 2 so positive weights invoke MAP")
    if args.beam_size < 1 or args.branching_factor < 1:
        raise SystemExit("--beam-size and --branching-factor must be positive")
    if args.transversal_groups < 1:
        raise SystemExit("--transversal-groups must be positive")
    if args.beam_size % args.transversal_groups:
        raise SystemExit("--beam-size must be divisible by --transversal-groups")
    if not args.model_path and args.branching_factor >= 64:
        raise SystemExit("tiny-model --branching-factor must be less than its 64-token vocabulary")

    checkpoint_sampler = None
    checkpoint_model = None
    if args.model_path:
        checkpoint_model, checkpoint_sampler = _load_checkpoint(args, device)

    cases: list[dict[str, Any]] = []
    for seed in args.seeds:
        if checkpoint_model is None:
            model = _tiny_model(
                seed,
                device,
                max_sequence_length=args.generation_length + 8,
            )
            if args.compile_model:
                # pyrefly: ignore [bad-assignment]
                model: torch.nn.Module = torch.compile(model, dynamic=True)
            prompt = _tiny_prompt(seed, device)
            eos_token_ids: tuple[int, ...] = ()
            prompt_label = [int(token) for token in prompt[0].tolist()]
        else:
            assert checkpoint_sampler is not None
            model = checkpoint_model
            prompt_text = _checkpoint_prompt(seed)
            prompt = checkpoint_sampler._preprocess_prompt(prompt_text)
            eos_token_ids = checkpoint_sampler._eos_token_ids()
            prompt_label = prompt_text
        attention = torch.ones_like(prompt)

        for layout in args.layouts:
            num_groups = args.transversal_groups if layout == "transversal" else 1
            with _autocast_context(device, not args.no_autocast):
                baseline = left_to_right_beam_sample(
                    model,
                    prompt,
                    attention,
                    generation_length=args.generation_length,
                    beam_size=args.beam_size,
                    branching_factor=args.branching_factor,
                    eos_token_ids=eos_token_ids,
                    num_groups=num_groups,
                )

            for weight in args.weights:
                case = _run_case(
                    model=model,
                    prompt=prompt,
                    attention=attention,
                    eos_token_ids=eos_token_ids,
                    generation_length=args.generation_length,
                    beam_size=args.beam_size,
                    branching_factor=args.branching_factor,
                    num_groups=num_groups,
                    weight=weight,
                    baseline=baseline,
                    device=device,
                    autocast=not args.no_autocast,
                )
                case.update({"seed": seed, "layout": layout, "prompt": prompt_label})
                cases.append(case)

    summaries = _summaries(cases)
    _print_summaries(summaries)
    payload = {
        "mode": "checkpoint" if args.model_path else "tiny",
        "model_path": args.model_path,
        "device": str(device),
        "compile_model": args.compile_model,
        "autocast": not args.no_autocast,
        "seeds": args.seeds,
        "weights": args.weights,
        "layouts": args.layouts,
        "generation_length": args.generation_length,
        "beam_size": args.beam_size,
        "branching_factor": args.branching_factor,
        "transversal_groups": args.transversal_groups,
        "summaries": summaries,
        "cases": cases,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""Micro-benchmark: the selection step is not resolvable on CUDA at our problem size.

Reviewer question: "what portion of the efficiency gain comes from DPP selection, given
Table 3 shows the selection step is slower than the simple baseline?"

This script supports the paper's claim that at the sizes we actually run
(6 groups x 10 per GPU on 2 GPUs -> B=120 candidates, k=12 selections) the *index
selection primitive* costs are below the measurement floor of CUDA event timing:
every variant is a handful of kernel launches on <=120 floats, so the timing is
launch-bound, not work-bound.

Method (this is the part that makes the claim defensible):
  * R independent repeats of N timed iterations each -> distribution over repeat means.
  * A `null` case (no device work) measures the harness/launch floor.
  * `arange` and `arange_again` are the SAME op timed twice -> gives the run-to-run
    spread you would see even for a zero-difference comparison. Any inter-variant
    difference smaller than this is not measurable.
  * Verdict block compares the largest inter-variant gap against that null gap.
  * --sweep shows where the variants *do* separate (large B), i.e. the size at which
    the comparison would become meaningful.

Run:  python bench_selection_step.py                 # main table + verdict
      python bench_selection_step.py --sweep         # + resolvability vs B
      python bench_selection_step.py --no-validate   # primitives only
"""

import argparse
import statistics
from collections.abc import Callable
from typing import Any

import torch


CaseFn = Callable[[], Any]


def time_repeats(fn: CaseFn, iters: int, repeats: int, warmup: int = 500) -> list[float]:
    """Return one mean-us-per-call figure per repeat (CUDA-event timed)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    out = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / iters)
    return out


def summarize(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        # 95% CI on the mean of repeat-means
        "ci": 1.96 * sd / (len(samples) ** 0.5) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def build_cases(  # noqa: C901
    B: int,
    k: int,
    ws: int,
    dev: torch.device,
    with_validate: bool,
) -> list[tuple[str, CaseFn]]:
    """Selection variants at the given size. Returns [(name, callable)]."""
    g = B // k
    local_B, local_k = B // ws, k // ws
    scores = torch.rand(B, device=dev)
    offsets = torch.arange(k, device=dev) * g
    rank_offsets = torch.arange(ws, device=dev) * local_B

    def null() -> None:
        return None

    def arange() -> torch.Tensor:
        return torch.arange(k, device=dev)

    def topk() -> torch.Tensor:
        return torch.topk(scores, k=k).indices

    def group_argmax() -> torch.Tensor:
        return torch.argmax(scores.view(k, g), dim=1) + offsets

    def per_rank_topk() -> torch.Tensor:
        out = []
        for offset in rank_offsets:
            out.append(torch.topk(scores[offset : offset + local_B], k=local_k).indices + offset)
        return torch.cat(out)

    # Mirrors BaseSelector._validate_global_selection: the .item() calls force a
    # device sync, which is the only part of the selection step with a real cost.
    def validate(ret: torch.Tensor, transversal: bool) -> bool:
        ret = ret.long()
        if ret.dim() != 1 or ret.numel() != k or ret.numel() == 0:
            return False
        if ret.min().item() < 0 or ret.max().item() >= B:
            return False
        if torch.unique(ret).numel() != k:
            return False
        if transversal:
            group_ids = torch.div(ret, g, rounding_mode="floor")
            return torch.unique(group_ids).numel() == k
        for rank in range(ws):
            start = rank * local_B
            cnt = ((ret >= start) & (ret < start + local_B)).sum().item()
            if cnt != local_k:
                return False
        return True

    cases: list[tuple[str, CaseFn]] = [
        ("null (no device work)", null),
        ("arange (baseline)", arange),
        ("arange_again (same op, noise probe)", arange),
        ("topk (quality-only, global)", topk),
        ("group_argmax (quality-only, transversal)", group_argmax),
        ("per_rank_topk (quality-only, distributed)", per_rank_topk),
    ]
    if with_validate:
        cases += [
            ("arange + validate", lambda: validate(arange(), transversal=False)),
            ("topk + validate", lambda: validate(topk(), transversal=False)),
            ("group_argmax + validate", lambda: validate(group_argmax(), transversal=True)),
            ("per_rank_topk + validate", lambda: validate(per_rank_topk(), transversal=False)),
        ]
    return cases


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=120, help="total candidates (batch_size * world_size)")
    p.add_argument("--k", type=int, default=12, help="selections (n_groups * world_size)")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--iters", type=int, default=2000, help="timed calls per repeat")
    p.add_argument("--repeats", type=int, default=20, help="independent repeats")
    p.add_argument("--no-validate", action="store_true", help="primitives only")
    p.add_argument("--sweep", action="store_true", help="also sweep B to find where variants separate")
    p.add_argument("--forward-ms", type=float, default=929.0, help="model forward for context (Table 3)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs cuda"
    dev = torch.device("cuda")
    B, k, ws = args.B, args.k, args.world_size
    assert B % k == 0, "B must be divisible by k (transversal group_size)"
    assert k % ws == 0, "k must be divisible by world_size"

    print(f"device = {torch.cuda.get_device_name(0)}   torch = {torch.__version__}")
    print(f"B={B}  k={k}  group_size={B // k}  world_size={ws}")
    print(f"timing = {args.repeats} repeats x {args.iters} iters, CUDA events\n")

    cases = build_cases(B, k, ws, dev, not args.no_validate)
    stats = {name: summarize(time_repeats(fn, args.iters, args.repeats)) for name, fn in cases}

    width = max(len(n) for n in stats)
    print(f"{'case'.ljust(width)}   mean us   +/-95%CI      sd     min     max")
    for name, s in stats.items():
        print(
            f"{name.ljust(width)}  {s['mean']:8.3f}  {s['ci']:8.3f}  {s['sd']:6.3f}  {s['min']:6.3f}  {s['max']:6.3f}",
        )

    # ---- verdict: is any inter-variant difference larger than same-op noise? ----
    noise = abs(stats["arange (baseline)"]["mean"] - stats["arange_again (same op, noise probe)"]["mean"])
    selection_names = [
        "arange (baseline)",
        "topk (quality-only, global)",
        "group_argmax (quality-only, transversal)",
        "per_rank_topk (quality-only, distributed)",
    ]
    means = {n: stats[n]["mean"] for n in selection_names}
    spread = max(means.values()) - min(means.values())
    floor = stats["null (no device work)"]["mean"]
    max_ci = max(stats[n]["ci"] for n in selection_names)

    print(f"\nharness/launch floor (null)            : {floor:.3f} us")
    print(f"same-op noise (arange vs arange_again) : {noise:.3f} us")
    print(f"widest 95% CI among selection variants : {max_ci:.3f} us")
    print(f"max spread across selection variants   : {spread:.3f} us")
    verdict = "NOT RESOLVABLE" if spread <= max(noise, max_ci) else "resolvable"
    print(f"-> inter-variant difference is {verdict} at this size")

    slowest = max(means, key=means.__getitem__)
    print(
        f"\nfor scale: slowest selection variant ({slowest.split(' ')[0]}) = {means[slowest]:.3f} us "
        f"= {100 * means[slowest] / (args.forward_ms * 1000):.6f}% of a {args.forward_ms:.0f} ms forward",
    )
    if not args.no_validate:
        val = stats["topk + validate"]["mean"] - stats["topk (quality-only, global)"]["mean"]
        print(f"validation ({'.item() syncs'}) adds {val:.3f} us on top of topk -- this is the measurable part")

    # ---- optional: where does the comparison become meaningful? ----
    if args.sweep:
        print("\nresolvability vs B (k = B/10, world_size=2):")
        print(f"{'B':>9}  {'arange':>9}  {'topk':>9}  {'grp_argmax':>11}  {'spread':>8}  {'noise':>8}  verdict")
        for Bs in (120, 1_200, 12_000, 120_000, 1_200_000):
            ks = Bs // 10
            ks -= ks % ws
            sub = dict(build_cases(Bs, ks, ws, dev, with_validate=False))
            it = max(50, args.iters // (Bs // 120))
            rep = max(5, args.repeats // 2)
            s = {n: summarize(time_repeats(f, it, rep, warmup=50)) for n, f in sub.items()}
            names = [
                "arange (baseline)",
                "topk (quality-only, global)",
                "group_argmax (quality-only, transversal)",
            ]
            m = [s[n]["mean"] for n in names]
            sp = max(m) - min(m)
            nz = abs(s["arange (baseline)"]["mean"] - s["arange_again (same op, noise probe)"]["mean"])
            tag = "not resolvable" if sp <= nz else "resolvable"
            print(f"{Bs:>9}  {m[0]:9.3f}  {m[1]:9.3f}  {m[2]:11.3f}  {sp:8.3f}  {nz:8.3f}  {tag}")

    for name, fn in cases[1:6]:
        out = fn()
        assert isinstance(out, torch.Tensor), name
        assert out.numel() == k and (out >= 0).all() and (out < B).all(), name
    print("\nshapes/ranges ok")


if __name__ == "__main__":
    main()

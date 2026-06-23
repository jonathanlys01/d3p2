"""Benchmark entrypoint for the experimental distributed MAP-DPP sampler."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from distributed_map.config import DistributedMAPConfig
from distributed_map.dummy_data import make_dummy_embeddings
from distributed_map.sampler import DistributedMAPSampler


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=1024, dest="sequence_length")
    parser.add_argument("--D", type=int, default=768, dest="embedding_dim")
    parser.add_argument("--d", type=int, default=128, dest="projected_dim")
    parser.add_argument("--L", type=int, default=32, dest="selections")
    parser.add_argument("--S", type=int, default=64, dest="local_trajectories")
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-triton", action="store_true")
    parser.add_argument("--dummy", action="store_true", help="Use deterministic dummy embeddings; required for now.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dummy:
        raise SystemExit("distributed_map only supports --dummy inputs in this experimental version")

    config = DistributedMAPConfig(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        projected_dim=args.projected_dim,
        selections=args.selections,
        local_trajectories=args.local_trajectories,
        block_n=args.block_n,
        seed=args.seed,
        projection_seed=args.projection_seed,
        device=args.device,
        require_triton=args.require_triton,
    )
    sampler = DistributedMAPSampler(config)

    try:
        embeddings = make_dummy_embeddings(config, device=sampler.device)
        for _ in range(args.warmup):
            _ = sampler.sample(embeddings)
            _sync(sampler.device)

        timings_ms: list[float] = []
        for _ in range(args.repeat):
            _sync(sampler.device)
            start = time.perf_counter()
            result = sampler.sample(embeddings)
            _sync(sampler.device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

        if sampler.rank == 0:
            print(
                "distributed_map benchmark: "
                f"world_size={sampler.world_size}, S={config.local_trajectories}, "
                f"N={config.sequence_length}, D={config.embedding_dim}, "
                f"d={config.projected_dim}, L={config.selections}",
            )
            print(f"device={sampler.device}, used_triton={result.used_triton}, winner_rank={result.winner_rank}")
            print(
                f"mean_ms={statistics.mean(timings_ms):.3f}, "
                f"median_ms={statistics.median(timings_ms):.3f}, "
                f"min_ms={min(timings_ms):.3f}, max_ms={max(timings_ms):.3f}",
            )
            print(f"score={float(result.score.item()):.6f}")
            print("selected=" + " ".join(str(int(x)) for x in result.selected.cpu().tolist()))
    finally:
        sampler.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark script to compare dispatch_sequences implementations.

This script compares:
1. Original implementation (using all_gather_object with pickle)
2. Optimized implementation (using all_gather_into_tensor)
3. Hybrid implementation (fast path for uniform sizes)

Run with: torchrun --nproc_per_node=<N_GPUS> benchmark_dispatch.py
"""

import statistics
import time
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# ORIGINAL IMPLEMENTATION (all_gather_object - slow pickle-based)
# ============================================================================


def dispatch_sequences_original(
    seq_ids: Optional[torch.Tensor],
    batch_size: int,
    world_size: int,
    rank: int,
    last: bool = False,
) -> torch.Tensor:
    """Original implementation using all_gather_object."""
    gather_indices = [None for _ in range(world_size)]

    if seq_ids is not None:
        seq_ids = seq_ids.to(dtype=torch.int32, device="cuda")

    dist.all_gather_object(gather_indices, seq_ids)

    all_indices_: list[torch.Tensor] = [idx.to("cuda") for idx in gather_indices if idx is not None]

    # Handle case where all ranks have None
    if not all_indices_:
        return torch.empty((0,), dtype=torch.int32, device="cuda")

    all_indices = torch.cat(all_indices_, dim=0)

    if last:
        return all_indices

    rank_indices = all_indices[rank * batch_size : (rank + 1) * batch_size]
    return rank_indices


# ============================================================================
# OPTIMIZED IMPLEMENTATION (all_gather_into_tensor - simplified)
# ============================================================================


def dispatch_sequences_optimized(
    seq_ids: Optional[torch.Tensor],
    batch_size: int,
    world_size: int,
    rank: int,
    last: bool = False,
) -> torch.Tensor:
    """Optimized implementation matching the simplified utils.py version."""
    # Determine local size
    local_size = seq_ids.size(0) if seq_ids is not None else 0

    # Gather sizes from all ranks
    sizes = torch.tensor([local_size], dtype=torch.int32, device="cuda")
    all_sizes = torch.zeros((world_size,), dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(all_sizes, sizes)

    # Handle empty case
    total_size = all_sizes.sum().item()
    if total_size == 0:
        return torch.empty((0,), dtype=torch.int32, device="cuda")

    # Pad to max size and gather
    max_size = int(all_sizes.max().item())

    if local_size == 0 or seq_ids is None:
        local_data = torch.full((max_size,), -1, dtype=torch.int32, device="cuda")
    elif local_size < max_size:
        local_data = seq_ids.to(dtype=torch.int32, device="cuda")
        pad_size = max_size - local_size
        padding = torch.full((pad_size,), -1, dtype=torch.int32, device="cuda")
        local_data = torch.cat([local_data, padding], dim=0)
    else:
        local_data = seq_ids.to(dtype=torch.int32, device="cuda")

    # Gather all data
    buffer_size = world_size * max_size
    gather_buffer = torch.zeros((buffer_size,), dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(gather_buffer, local_data)

    # Filter out sentinel values
    all_indices = gather_buffer[gather_buffer != -1]

    if last:
        return all_indices

    # Slice for this rank
    rank_indices = all_indices[rank * batch_size : (rank + 1) * batch_size]
    return rank_indices


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def benchmark_implementation(  # noqa: PLR0913
    impl_name: str,
    impl_func,
    seq_ids: Optional[torch.Tensor],
    batch_size: int,
    world_size: int,
    rank: int,
    n_iterations: int = 100,
    warmup: int = 10,
    **kwargs,
) -> dict:
    """Benchmark a single implementation."""

    # Warmup
    for _ in range(warmup):
        _ = impl_func(seq_ids, batch_size, world_size, rank, **kwargs)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = impl_func(seq_ids, batch_size, world_size, rank, **kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        "name": impl_name,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def run_benchmarks(rank: int, world_size: int, batch_size: int = 32):
    """Run all benchmarks and report results."""

    print(f"[Rank {rank}] Starting benchmarks...")
    print(f"[Rank {rank}] World size: {world_size}, Batch size: {batch_size}")

    # Calculate variable sizes that sum to world_size * batch_size
    # Create a distribution where some ranks have more, some have less
    total_sequences = world_size * batch_size
    base_per_rank = batch_size

    # Create variable distribution: rank 0 gets more, last rank gets less
    variable_sizes = []
    remaining = total_sequences
    for r in range(world_size):
        if r < world_size - 1:
            # Vary between 0.5x and 1.5x the base, but ensure we don't go negative
            variation = int(base_per_rank * (0.5 + (r % 3) * 0.5))
            size = min(variation, remaining - (world_size - r - 1))  # Leave room for remaining ranks
            size = max(0, size)  # Ensure non-negative
        else:
            # Last rank gets whatever is left
            size = remaining
        variable_sizes.append(size)
        remaining -= size

    # Test scenarios
    scenarios = [
        {
            "name": "Uniform size (batch_size)",
            "seq_ids": torch.randint(0, 10000, (batch_size,), device="cuda"),
        },
        {
            "name": "Small uniform size (batch_size // 2)",
            "seq_ids": torch.randint(0, 10000, (batch_size // 2,), device="cuda"),
        },
        {
            "name": f"Variable size (total={total_sequences}, sizes={variable_sizes})",
            "seq_ids": torch.randint(0, 10000, (variable_sizes[rank],), dtype=torch.int32, device="cuda"),
        },
    ]

    results = []

    for scenario in scenarios:
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Scenario: {scenario['name']}")
            print(f"{'=' * 80}")

        dist.barrier()

        seq_ids: torch.Tensor = scenario["seq_ids"]  # type: ignore

        # Benchmark original implementation
        result_original = benchmark_implementation(
            "Original (all_gather_object)",
            dispatch_sequences_original,
            seq_ids,
            batch_size,
            world_size,
            rank,
        )

        # Benchmark optimized implementation
        result_optimized = benchmark_implementation(
            "Optimized (all_gather_into_tensor)",
            dispatch_sequences_optimized,
            seq_ids,
            batch_size,
            world_size,
            rank,
        )

        results.append(
            {
                "scenario": scenario["name"],
                "original": result_original,
                "optimized": result_optimized,
            },
        )

        # Print results from rank 0
        if rank == 0:
            print(f"\n{scenario['name']}:")
            print(f"  Original:  {result_original['mean_ms']:.4f} ms (± {result_original['stdev_ms']:.4f})")
            print(f"  Optimized: {result_optimized['mean_ms']:.4f} ms (± {result_optimized['stdev_ms']:.4f})")

            speedup = result_original["mean_ms"] / result_optimized["mean_ms"]
            print(f"\n  Speedup (Optimized vs Original): {speedup:.2f}x")

            if speedup > 1:
                improvement = (1 - 1 / speedup) * 100
                print(f"  Performance improvement: {improvement:.1f}% faster")
            else:
                regression = (speedup - 1) * 100
                print(f"  Performance regression: {regression:.1f}% slower")

        dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("Benchmark complete!")
        print(f"{'=' * 80}")


def main():
    """Main entry point."""
    rank, world_size, local_rank = setup_distributed()

    try:
        # Run benchmarks with different batch sizes
        for batch_size in [16, 32, 64]:
            if rank == 0:
                print(f"\n\n{'#' * 80}")
                print(f"# BATCH SIZE: {batch_size}")
                print(f"{'#' * 80}")

            dist.barrier()
            run_benchmarks(rank, world_size, batch_size)
            dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

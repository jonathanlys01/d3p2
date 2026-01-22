#!/usr/bin/env python3
"""
Communication Health Test script to verify NCCL and Torch distributed setup.
Run with: torchrun --nproc_per_node=gpu tests/comm_health.py
"""

import os
import time

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        # Look for environment variables set by torchrun/slurm
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        dist.init_process_group(backend="nccl")
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_all_reduce(rank, world_size):
    """Test all_reduce by summing ones across all ranks."""
    if rank == 0:
        print("\n--- Testing all_reduce ---")

    # Create a tensor of ones
    tensor = torch.ones(1, device="cuda")

    dist.barrier()
    start_time = time.perf_counter()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    # Check result
    expected = float(world_size)
    actual = tensor.item()

    status = "PASSED" if actual == expected else f"FAILED (Expected {expected}, got {actual})"

    if rank == 0:
        print(f"all_reduce: {status}")
        print(f"Latency: {latency_ms:.4f} ms")

    return status == "PASSED"


def test_all_gather(rank, world_size):
    """Test all_gather by gathering rank IDs from all ranks."""
    if rank == 0:
        print("\n--- Testing all_gather ---")

    # Each rank provides its own rank as a tensor
    local_tensor = torch.tensor([rank], dtype=torch.int32, device="cuda")
    # Buffer to hold gathered results
    gather_list = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(world_size)]

    dist.barrier()
    start_time = time.perf_counter()

    dist.all_gather(gather_list, local_tensor)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    # Check result
    gathered_ranks = [t.item() for t in gather_list]
    expected_ranks = list(range(world_size))

    if sorted(gathered_ranks) == expected_ranks:
        status = "PASSED"
    else:
        status = f"FAILED (Expected {expected_ranks}, got {gathered_ranks})"

    if rank == 0:
        print(f"all_gather: {status}")
        print(f"Latency: {latency_ms:.4f} ms")
        print(f"Gathered Ranks: {gathered_ranks}")

    return status == "PASSED"


def test_all_gather_into_tensor(rank, world_size):
    """Test all_gather_into_tensor (more efficient than all_gather)."""
    if rank == 0:
        print("\n--- Testing all_gather_into_tensor ---")

    local_tensor = torch.tensor([rank], dtype=torch.int32, device="cuda")
    output_tensor = torch.zeros(world_size, dtype=torch.int32, device="cuda")

    dist.barrier()
    start_time = time.perf_counter()

    dist.all_gather_into_tensor(output_tensor, local_tensor)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    # Check result
    gathered_ranks = output_tensor.tolist()
    expected_ranks = list(range(world_size))

    if sorted(gathered_ranks) == expected_ranks:
        status = "PASSED"
    else:
        status = f"FAILED (Expected {expected_ranks}, got {gathered_ranks})"

    if rank == 0:
        print(f"all_gather_into_tensor: {status}")
        print(f"Latency: {latency_ms:.4f} ms")

    return status == "PASSED"


def main():
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("Communication Health Check")
        print(f"World Size: {world_size}")
        print(f"Backend: {dist.get_backend()}")

    try:
        results = []
        results.append(test_all_reduce(rank, world_size))
        results.append(test_all_gather(rank, world_size))
        results.append(test_all_gather_into_tensor(rank, world_size))

        dist.barrier()

        if rank == 0:
            if all(results):
                print("\n" + "=" * 40)
                print("COMMUNICATION HEALTH: EXCELLENT")
                print("=" * 40)
            else:
                print("\n" + "!" * 40)
                print("COMMUNICATION HEALTH: ISSUES DETECTED")
                print("!" * 40)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

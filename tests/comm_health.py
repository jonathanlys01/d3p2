# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "nvidia-ml-py",
# ]
# ///

import os
import time

import pynvml
import torch
import torch.distributed as dist


def run_diagnostics(duration_seconds=600, matrix_size=8192):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize NVML and get the handle for this specific GPU
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    if local_rank == 0:
        print(f"Starting {duration_seconds}s diagnostic loop on {world_size} GPUs...")

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
        dist.all_reduce(C)
    torch.cuda.synchronize()

    start_time = time.time()
    end_time = start_time + duration_seconds

    iter_count = 0
    total_matmul_time = 0.0
    total_nccl_time = 0.0

    power_readings = []

    while time.time() < end_time:
        # 1. Individual Performance (Tensor Cores)
        t0 = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        t1 = time.time()

        # 2. Synchronized Communication (NCCL)
        t2 = time.time()
        dist.all_reduce(C, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t3 = time.time()

        total_matmul_time += t1 - t0
        total_nccl_time += t3 - t2

        # 3. Power Measurement (Sampled every 10 iterations)
        if iter_count % 10 == 0:
            # NVML returns power in milliwatts, convert to Watts
            power_mW = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
            power_readings.append(power_mW / 1000.0)

        iter_count += 1

    # Math for metrics
    flops_per_matmul = 2 * (matrix_size**3)
    data_size_bytes = C.numel() * C.element_size()

    avg_matmul_time = total_matmul_time / iter_count
    avg_nccl_time = total_nccl_time / iter_count

    tflops = (flops_per_matmul / avg_matmul_time) / 1e12
    alg_bw = (data_size_bytes / avg_nccl_time) / 1e9
    bus_bw = alg_bw * (2 * (world_size - 1) / world_size)

    # Calculate Power Stats
    avg_power = sum(power_readings) / len(power_readings)
    max_power = max(power_readings)

    print(
        f"[GPU {local_rank}] Matmul: {tflops:.2f} TFLOPS | NCCL Bus BW: {bus_bw:.2f} GB/s | Power: {avg_power:.0f}W avg / {max_power:.0f}W max"
    )

    # Cleanup
    pynvml.nvmlShutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_diagnostics(duration_seconds=600)

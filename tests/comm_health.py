"""GPU cluster health diagnostic.

Measures per-GPU compute (TFLOPS), collective communication bandwidth (NCCL
all_reduce), point-to-point bandwidth between every GPU pair, and power draw.
Because all GPUs are the same model, every metric is normalised against the
cluster median so that outliers are immediately visible.

Usage (example – 8 GPUs, 60 s run):
    torchrun --nproc_per_node=8 tests/comm_health.py --duration_seconds 60
"""

import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import pynvml
import torch
import torch.distributed as dist


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _deviation_pct(value: float, reference: float) -> float:
    """Percentage deviation from reference (positive = better, negative = worse)."""
    if reference == 0:
        return 0.0
    return (value - reference) / reference * 100.0


def _flag(dev_pct: float, warn_thresh: float = -5.0, bad_thresh: float = -10.0) -> str:
    """Return an emoji flag based on deviation from reference."""
    if dev_pct < bad_thresh:
        return "🔴"
    if dev_pct < warn_thresh:
        return "🟡"
    return "🟢"


@dataclass
class _PowerStats:
    avg_watts: float
    peak_watts: float
    floor_watts: float
    sample_count: int


class _PowerSampler:
    """Sample NVML power in the background while a workload is active."""

    def __init__(self, nvml_handle: Any, interval_seconds: float = 0.05) -> None:
        self._nvml_handle = nvml_handle
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[float] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0)
            time.sleep(self._interval_seconds)

    def __enter__(self) -> "_PowerSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds * 4)


def _summarize_power(samples: list[float]) -> _PowerStats:
    if not samples:
        return _PowerStats(avg_watts=0.0, peak_watts=0.0, floor_watts=0.0, sample_count=0)
    return _PowerStats(
        avg_watts=sum(samples) / len(samples),
        peak_watts=max(samples),
        floor_watts=min(samples),
        sample_count=len(samples),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────


def _benchmark_matmul(
    device: torch.device,
    nvml_handle: Any,
    matrix_size: int,
    duration_seconds: float,
    warmup_iters: int = 10,
) -> tuple[float, float, _PowerStats]:
    """Return (avg_tflops, cv_tflops, load_power_stats)."""
    A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    for _ in range(warmup_iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    flops = 2 * (matrix_size**3)
    per_iter: list[float] = []
    end = time.time() + duration_seconds

    with _PowerSampler(nvml_handle) as power_sampler:
        while time.time() < end:
            t0 = time.time()
            torch.matmul(A, B)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            per_iter.append((flops / elapsed) / 1e12)

    avg = sum(per_iter) / len(per_iter)
    cv = _stdev(per_iter, avg) / avg if avg > 0 else 0.0
    return avg, cv, _summarize_power(power_sampler.samples)


def _benchmark_allreduce(
    tensor: torch.Tensor,
    nvml_handle: Any,
    world_size: int,
    duration_seconds: float,
    warmup_iters: int = 10,
) -> tuple[float, float, _PowerStats]:
    """Return (avg_bus_bw_GBps, cv_bus_bw, load_power_stats)."""
    for _ in range(warmup_iters):
        dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    data_bytes = tensor.numel() * tensor.element_size()
    per_iter: list[float] = []
    end = time.time() + duration_seconds

    with _PowerSampler(nvml_handle) as power_sampler:
        while time.time() < end:
            buf = tensor.clone()
            t0 = time.time()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            alg_bw = (data_bytes / elapsed) / 1e9
            bus_bw = alg_bw * (2 * (world_size - 1) / world_size)
            per_iter.append(bus_bw)

    avg = sum(per_iter) / len(per_iter)
    cv = _stdev(per_iter, avg) / avg if avg > 0 else 0.0
    return avg, cv, _summarize_power(power_sampler.samples)


def _explore_power_envelope(
    device: torch.device,
    nvml_handle: Any,
    matrix_size: int,
    step_seconds: float,
) -> _PowerStats:
    """Ramp a matmul workload to expose sustained and peak draw under load."""
    step_sizes = sorted(
        {
            max(1024, matrix_size // 4),
            max(1024, matrix_size // 2),
            max(2048, (matrix_size * 3) // 4),
            matrix_size,
        },
    )
    all_samples: list[float] = []

    for step_size in step_sizes:
        A = torch.randn(step_size, step_size, dtype=torch.float16, device=device)
        B = torch.randn(step_size, step_size, dtype=torch.float16, device=device)
        for _ in range(4):
            torch.matmul(A, B)
        torch.cuda.synchronize()

        end = time.time() + step_seconds
        with _PowerSampler(nvml_handle) as power_sampler:
            while time.time() < end:
                torch.matmul(A, B)
                torch.cuda.synchronize()
        all_samples.extend(power_sampler.samples)

    return _summarize_power(all_samples)


def _run_p2p_pair(
    buf: torch.Tensor,
    local_rank: int,
    peer: int,
    iters: int,
    n_bytes: int,
) -> float:
    """Execute one timed send/recv pair; return measured BW in GB/s (0 for receiver)."""
    # Warmup
    for _ in range(5):
        if local_rank < peer:
            dist.send(buf, dst=peer)
        else:
            dist.recv(buf, src=peer)
    # Timed pass
    if local_rank < peer:
        t0 = time.time()
        for _ in range(iters):
            dist.send(buf, dst=peer)
        torch.cuda.synchronize()
        return (n_bytes * iters / (time.time() - t0)) / 1e9
    for _ in range(iters):
        dist.recv(buf, src=peer)
    return 0.0


def _benchmark_p2p(
    device: torch.device,
    local_rank: int,
    world_size: int,
    tensor_size_mb: int = 512,
    iters: int = 20,
) -> dict[tuple[int, int], float]:
    """Return a dict mapping (src, dst) -> bandwidth_GBps.

    Only the sender measures; results are shared via all_reduce so every rank
    ends up with the full matrix.  Complexity stays O(N) per rank, O(N²) aggregated.
    """
    n_bytes = tensor_size_mb * 1024 * 1024
    n_elems = n_bytes // 2  # float16 → 2 bytes each
    buf = torch.zeros(n_elems, dtype=torch.float16, device=device)

    # results[src * world_size + dst] stores BW; only the sender writes a non-zero value.
    results = torch.zeros(world_size * world_size, dtype=torch.float32, device=device)

    for peer in range(world_size):
        if peer == local_rank:
            continue
        bw = _run_p2p_pair(buf, local_rank, peer, iters, n_bytes)
        if bw > 0:
            results[local_rank * world_size + peer] = bw

    # Each rank broadcasts its measurements to all others
    dist.all_reduce(results, op=dist.ReduceOp.SUM)

    bw_map: dict[tuple[int, int], float] = {}
    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                continue
            bw = results[src * world_size + dst].item()
            if bw > 0:
                bw_map[(src, dst)] = bw
                bw_map[(dst, src)] = bw  # store symmetric entry for lookup convenience

    return bw_map


# ──────────────────────────────────────────────────────────────────────────────
# Reporting (rank-0 only)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _ClusterStats:
    tflops: list[float]
    bus_bw: list[float]
    cv_bw: list[float]
    compute_power: list[float]
    comm_power: list[float]
    peak_power: list[float]


def _print_per_gpu_table(  # noqa: PLR0913
    world_size: int,
    tflops_list: list[float],
    bus_bw_list: list[float],
    cv_bw_list: list[float],
    compute_power_list: list[float],
    comm_power_list: list[float],
    peak_power_list: list[float],
    warn_thresh: float,
    bad_thresh: float,
) -> list[int]:
    """Print the per-GPU metrics table and return a list of anomalous rank IDs."""
    med_tflops = _median(tflops_list)
    med_bus_bw = _median(bus_bw_list)
    med_compute_power = _median(compute_power_list)
    med_comm_power = _median(comm_power_list)
    med_peak_power = _median(peak_power_list)

    print(f"\n{'=' * 70}")
    print("  Per-GPU Metrics  (deviation from cluster median)")
    print(
        "  Median → "
        f"{med_tflops:.2f} TFLOPS | {med_bus_bw:.2f} GB/s bus BW | "
        f"{med_compute_power:.1f} W compute | {med_peak_power:.1f} W peak",
    )
    print(f"{'─' * 70}")
    hdr = (
        f"  {'GPU':>3}  {'TFLOPS':>8}  {'Dev%':>6}"
        f"  {'BusBW(GB/s)':>11}  {'Dev%':>6}"
        f"  {'CV_bw':>5}  {'CmpPwr':>7}  {'ComPwr':>7}  {'PeakW':>7}  Flags"
    )
    print(hdr)
    print(f"{'─' * 70}")

    anomalous: list[int] = []
    for rank in range(world_size):
        tf = tflops_list[rank]
        bw = bus_bw_list[rank]
        cv = cv_bw_list[rank]
        compute_pw = compute_power_list[rank]
        comm_pw = comm_power_list[rank]
        peak_pw = peak_power_list[rank]

        dev_tf = _deviation_pct(tf, med_tflops)
        dev_bw = _deviation_pct(bw, med_bus_bw)
        dev_compute_pw = _deviation_pct(compute_pw, med_compute_power)
        dev_comm_pw = _deviation_pct(comm_pw, med_comm_power)
        dev_peak_pw = _deviation_pct(peak_pw, med_peak_power)

        f_tf = _flag(dev_tf, warn_thresh, bad_thresh)
        f_bw = _flag(dev_bw, warn_thresh, bad_thresh)
        f_pw = _flag(min(dev_compute_pw, dev_comm_pw, dev_peak_pw), warn_thresh, bad_thresh)

        flags = f"compute:{f_tf} comm:{f_bw} power:{f_pw}"
        row = (
            f"  {rank:>3}  {tf:>8.2f}  {dev_tf:>+6.1f}%"
            f"  {bw:>11.2f}  {dev_bw:>+6.1f}%"
            f"  {cv:>5.3f}  {compute_pw:>7.1f}  {comm_pw:>7.1f}  {peak_pw:>7.1f}  {flags}"
        )
        print(row)

        if (
            dev_tf < bad_thresh
            or dev_bw < bad_thresh
            or (dev_peak_pw < bad_thresh and (dev_tf < warn_thresh or dev_bw < warn_thresh))
        ):
            anomalous.append(rank)

    print(f"{'─' * 70}")
    print(
        "  CV_bw = coefficient of variation of per-iteration all_reduce BW (higher → more jitter)",
    )
    return anomalous


def _print_p2p_table(
    world_size: int,
    p2p_bw: dict[tuple[int, int], float],
    warn_thresh: float,
    bad_thresh: float,
) -> list[tuple[int, int, float]]:
    """Print the p2p bandwidth matrix and return a list of slow (src, dst, bw) tuples."""
    print(f"\n{'=' * 70}")
    print("  Point-to-Point Bandwidth (GB/s)  [sender ↔ receiver]")
    print(f"{'─' * 70}")

    pair_bws = [bw for (src, dst), bw in p2p_bw.items() if src < dst]
    med_p2p = _median(pair_bws) if pair_bws else 0.0
    slow_links: list[tuple[int, int, float]] = []

    bw_matrix: list[list[str]] = [["  —  "] * world_size for _ in range(world_size)]
    for (src, dst), bw in p2p_bw.items():
        if src < dst:
            dev = _deviation_pct(bw, med_p2p)
            flag = _flag(dev, warn_thresh, bad_thresh)
            bw_matrix[src][dst] = f"{bw:5.1f}{flag}"
            if dev < bad_thresh:
                slow_links.append((src, dst, bw))

    col_w = 9
    header_row = "  " + "src\\dst".ljust(5)
    for dst in range(world_size):
        header_row += f"  GPU{dst:>2}".ljust(col_w)
    print(header_row)

    for src in range(world_size):
        row = f"  GPU{src:<2} "
        for dst in range(world_size):
            if src == dst:
                row += "    —    "
            elif src < dst:
                row += f" {bw_matrix[src][dst]:<8}"
            else:
                row += f" {bw_matrix[dst][src]:<8}"
        print(row)

    print(f"{'─' * 70}")
    print(f"  Cluster median p2p: {med_p2p:.1f} GB/s")
    return slow_links


def _print_summary(  # noqa: C901, PLR0913
    anomalous_gpus: list[int],
    slow_links: list[tuple[int, int, float]],
    cv_bw_list: list[float],
    tflops_list: list[float],
    bus_bw_list: list[float],
    compute_power_list: list[float],
    comm_power_list: list[float],
    peak_power_list: list[float],
    med_p2p: float,
    warn_thresh: float,
    bad_thresh: float,
) -> None:
    """Print the diagnosis summary section."""
    print(f"\n{'=' * 70}")
    print("  DIAGNOSIS SUMMARY")
    print(f"{'─' * 70}")

    if not anomalous_gpus and not slow_links:
        print("  🟢 All GPUs healthy. No anomalies detected.")
    else:
        if anomalous_gpus:
            unique = sorted(set(anomalous_gpus))
            print(f"  🔴 Potentially faulty GPUs: {unique}")
            print(f"     ↳ These deviate >{-bad_thresh:.0f}% from median on compute or comms.")
        if slow_links:
            print("  🔴 Slow NVLink/PCIe links:")
            for src, dst, bw in slow_links:
                dev = _deviation_pct(bw, med_p2p)
                print(f"     ↳ GPU {src} ↔ GPU {dst}: {bw:.1f} GB/s ({dev:+.1f}% vs median)")

    med_tf = _median(tflops_list)
    med_bw = _median(bus_bw_list)
    med_compute_power = _median(compute_power_list)
    med_comm_power = _median(comm_power_list)
    med_peak_power = _median(peak_power_list)

    underpowered: list[int] = []
    normal_power_but_slow: list[int] = []
    for rank, (tf, bw, cpw, mpw, ppw) in enumerate(
        zip(tflops_list, bus_bw_list, compute_power_list, comm_power_list, peak_power_list, strict=True),
    ):
        slow_compute = _deviation_pct(tf, med_tf) < warn_thresh
        slow_comm = _deviation_pct(bw, med_bw) < warn_thresh
        low_power = (
            min(
                _deviation_pct(cpw, med_compute_power),
                _deviation_pct(mpw, med_comm_power),
                _deviation_pct(ppw, med_peak_power),
            )
            < warn_thresh
        )
        if (slow_compute or slow_comm) and low_power:
            underpowered.append(rank)
        elif (slow_compute or slow_comm) and not low_power:
            normal_power_but_slow.append(rank)

    if underpowered:
        print(f"  🟡 Likely underpowered GPUs: {underpowered}")
        print("     ↳ These are slower than peers and also fail to reach comparable load/peak wattage.")
    if normal_power_but_slow:
        print(f"  🟡 Slow despite normal power: {normal_power_but_slow}")
        print("     ↳ These draw expected power under load, so investigate clocks, thermals, or link issues.")

    high_cv = [r for r, cv in enumerate(cv_bw_list) if cv > 0.15]
    if high_cv:
        print(f"  🟡 GPUs with high all_reduce jitter (CV > 15%): {high_cv}")
        print("     ↳ These may act as stragglers and slow the entire collective.")

    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_diagnostics(
    duration_seconds: int = 60,
    matrix_size: int = 8192,
    p2p_size_mb: int = 512,
    warn_thresh: float = -5.0,
    bad_thresh: float = -10.0,
) -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    if local_rank == 0:
        gpu_name = pynvml.nvmlDeviceGetName(nvml_handle)
        print(f"\n{'=' * 70}")
        print("  GPU Cluster Health Diagnostic")
        print(f"  GPUs: {world_size} × {gpu_name}")
        print(f"  Matmul: {matrix_size}×{matrix_size} fp16 | Duration: {duration_seconds}s")
        print(f"{'=' * 70}\n")

    phase_dur = max(duration_seconds / 4, 1.0)

    # Phase 1: Compute (tensor cores)
    if local_rank == 0:
        print("[1/4] Benchmarking tensor-core compute (matmul with live power sampling)…")
    avg_tflops, cv_tflops, compute_power_stats = _benchmark_matmul(
        device,
        nvml_handle,
        matrix_size,
        phase_dur,
    )

    # Phase 2: Collective communication (NCCL all_reduce)
    if local_rank == 0:
        print("[2/4] Benchmarking collective communication (all_reduce with live power sampling)…")
    comm_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    avg_bus_bw, cv_bus_bw, comm_power_stats = _benchmark_allreduce(
        comm_tensor,
        nvml_handle,
        world_size,
        phase_dur,
    )

    # Phase 3: Power exploration
    if local_rank == 0:
        print("[3/4] Exploring power envelope with stepped matmul load…")
    explore_power_stats = _explore_power_envelope(
        device,
        nvml_handle,
        matrix_size,
        max(phase_dur / 4, 0.75),
    )

    # Phase 4: Point-to-point bandwidth
    if local_rank == 0:
        print("[4/4] Benchmarking point-to-point bandwidth…")
    p2p_bw = _benchmark_p2p(device, local_rank, world_size, tensor_size_mb=p2p_size_mb)

    # Gather all per-GPU scalars onto rank 0
    local_stats = torch.tensor(
        [
            avg_tflops,
            cv_tflops,
            avg_bus_bw,
            cv_bus_bw,
            compute_power_stats.avg_watts,
            comm_power_stats.avg_watts,
            max(compute_power_stats.peak_watts, comm_power_stats.peak_watts, explore_power_stats.peak_watts),
        ],
        dtype=torch.float32,
        device=device,
    )
    all_stats = [torch.zeros_like(local_stats) for _ in range(world_size)]
    dist.all_gather(all_stats, local_stats)

    if local_rank == 0:
        tflops_list = [s[0].item() for s in all_stats]
        bus_bw_list = [s[2].item() for s in all_stats]
        cv_bw_list = [s[3].item() for s in all_stats]
        compute_power_list = [s[4].item() for s in all_stats]
        comm_power_list = [s[5].item() for s in all_stats]
        peak_power_list = [s[6].item() for s in all_stats]

        anomalous = _print_per_gpu_table(
            world_size,
            tflops_list,
            bus_bw_list,
            cv_bw_list,
            compute_power_list,
            comm_power_list,
            peak_power_list,
            warn_thresh,
            bad_thresh,
        )
        slow_links = _print_p2p_table(world_size, p2p_bw, warn_thresh, bad_thresh)

        pair_bws = [bw for (src, dst), bw in p2p_bw.items() if src < dst]
        med_p2p = _median(pair_bws) if pair_bws else 0.0

        _print_summary(
            anomalous,
            slow_links,
            cv_bw_list,
            tflops_list,
            bus_bw_list,
            compute_power_list,
            comm_power_list,
            peak_power_list,
            med_p2p,
            warn_thresh,
            bad_thresh,
        )

    pynvml.nvmlShutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU cluster health diagnostic")
    parser.add_argument(
        "--duration_seconds",
        type=int,
        default=600,
        help="Total wall-clock duration (split across compute, comm, power exploration, and p2p phases)",
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=8192,
        help="N for NxN matmul (fp16)",
    )
    parser.add_argument(
        "--p2p_size_mb",
        type=int,
        default=512,
        help="Tensor size per p2p transfer in MB",
    )
    parser.add_argument(
        "--warn_thresh",
        type=float,
        default=-5.0,
        help="Warning threshold %% deviation below median (default: -5%%)",
    )
    parser.add_argument(
        "--bad_thresh",
        type=float,
        default=-10.0,
        help="Critical threshold %% deviation below median (default: -10%%)",
    )
    args = parser.parse_args()
    run_diagnostics(
        duration_seconds=args.duration_seconds,
        matrix_size=args.matrix_size,
        p2p_size_mb=args.p2p_size_mb,
        warn_thresh=args.warn_thresh,
        bad_thresh=args.bad_thresh,
    )

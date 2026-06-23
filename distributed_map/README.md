# Distributed MAP DPP Experimental Kernel

This folder is a standalone experiment for projected batched greedy MAP inference for DPPs. It is intentionally not wired into the existing `d5p4` selector factory.

The implementation uses only dummy embeddings. It does not load real model outputs or datasets.

## Shape Defaults

- `N=1024` candidate items
- `D=768` input embedding dimension
- `d=128` projected dimension
- `BLOCK_N=256`
- bf16 embeddings and projections
- fp32 Cholesky state and scores

## Running

CPU reference path:

```bash
uv run python -m distributed_map.benchmark --dummy --N 1024 --D 768 --d 128 --L 32 --S 64
```

CUDA/Triton path, when available:

```bash
uv run python -m distributed_map.benchmark --dummy --require-triton
```

Distributed NCCL path:

```bash
torchrun --nproc_per_node=2 -m distributed_map.benchmark --dummy --require-triton --S 64
```

For weak scaling, launch the same command with `--nproc_per_node` in `{1,2,4,8}` while keeping `--S` fixed.

## Constraints

- The sampler never materializes an `N x N` kernel matrix. It uses projected linear-kernel dot products blockwise.
- Step 0 is partition-local: each global trajectory starts from the argmax of its assigned contiguous partition.
- Final distributed selection communicates a scalar score and the selected sequence payload.
- The local CPU environment used during development had no CUDA and no importable Triton, so GPU occupancy and NCCL latency must be measured on target hardware.

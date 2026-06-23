"""High-level orchestration for the experimental distributed MAP-DPP sampler."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from distributed_map.config import DistributedMAPConfig
from distributed_map.dummy_data import make_dummy_embeddings
from distributed_map.kernels import (
    HAS_TRITON,
    KernelResult,
    partition_initial_items,
    run_reference_projected_greedy,
    run_triton_projected_greedy,
)
from distributed_map.projection import project_embeddings


@dataclass
class SampleResult:
    selected: torch.Tensor
    score: torch.Tensor
    x_tilde: torch.Tensor
    local_kernel: KernelResult
    local_best_index: int
    winner_rank: int
    used_triton: bool


class DistributedMAPSampler:
    """Standalone projected MAP-DPP sampler with optional NCCL final selection."""

    def __init__(self, config: DistributedMAPConfig):
        self.config = config
        self.rank, self.world_size, self.local_rank = self._runtime_info()
        self.device = self._resolve_device()
        self._maybe_init_process_group()

    def sample(self, embeddings: torch.Tensor | None = None) -> SampleResult:
        """Run dummy-data projected sampling and return the globally best sequence."""

        if embeddings is None:
            embeddings = make_dummy_embeddings(self.config, device=self.device)
        else:
            embeddings = embeddings.to(device=self.device)

        x_tilde = project_embeddings(embeddings, self.config)
        return self.sample_projected(x_tilde)

    def sample_projected(self, x_tilde: torch.Tensor) -> SampleResult:
        """Run local trajectories from pre-projected embeddings."""

        x_tilde = x_tilde.to(device=self.device, dtype=self.config.projection_dtype).contiguous()
        initial_items = partition_initial_items(
            x_tilde,
            self.config,
            rank=self.rank,
            world_size=self.world_size,
        )

        use_triton = x_tilde.device.type == "cuda" and HAS_TRITON
        if self.config.require_triton and not use_triton:
            raise RuntimeError("require_triton=True but CUDA/Triton is unavailable")

        if use_triton:
            local_kernel = run_triton_projected_greedy(x_tilde, initial_items, self.config)
        else:
            local_kernel = run_reference_projected_greedy(x_tilde, initial_items, self.config)

        local_best_index = self._local_best_index(local_kernel)
        local_score = local_kernel.scores[local_best_index]
        local_selected = local_kernel.selected[local_best_index]

        global_score, global_selected, winner_rank = self._distributed_best(local_score, local_selected)
        return SampleResult(
            selected=global_selected,
            score=global_score,
            x_tilde=x_tilde,
            local_kernel=local_kernel,
            local_best_index=local_best_index,
            winner_rank=winner_rank,
            used_triton=use_triton,
        )

    def cleanup(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _local_best_index(self, result: KernelResult) -> int:
        max_score = torch.max(result.scores)
        candidates = torch.where(result.scores == max_score)[0]
        if candidates.numel() == 1:
            return int(candidates.item())

        starts = result.selected[candidates, 0]
        min_start = torch.min(starts)
        tied = candidates[starts == min_start]
        return int(torch.min(tied).item())

    def _distributed_best(
        self,
        local_score: torch.Tensor,
        local_selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.world_size == 1 or not dist.is_available() or not dist.is_initialized():
            return local_score.detach().clone(), local_selected.detach().clone(), self.rank

        score = local_score.detach().clone().to(device=self.device, dtype=torch.float32)
        dist.all_reduce(score, op=dist.ReduceOp.MAX)

        is_score_winner = torch.isclose(local_score.to(torch.float32), score, rtol=0.0, atol=0.0)
        start_value = torch.tensor(
            int(local_selected[0].item()) if bool(is_score_winner.item()) else self.config.sequence_length,
            device=self.device,
        )
        dist.all_reduce(start_value, op=dist.ReduceOp.MIN)

        is_start_winner = bool(is_score_winner.item()) and int(local_selected[0].item()) == int(start_value.item())
        rank_value = torch.tensor(self.rank if is_start_winner else self.world_size, device=self.device)
        dist.all_reduce(rank_value, op=dist.ReduceOp.MIN)
        winner_rank = int(rank_value.item())

        payload = local_selected.detach().clone().to(device=self.device, dtype=torch.int64)
        if self.rank != winner_rank:
            payload.zero_()
        dist.broadcast(payload, src=winner_rank)
        return score, payload, winner_rank

    def _resolve_device(self) -> torch.device:
        requested = self.config.device
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def _maybe_init_process_group(self) -> None:
        if self.world_size <= 1 or dist.is_initialized():
            return
        if self.device.type != "cuda":
            raise RuntimeError("distributed_map uses NCCL and requires CUDA for distributed runs")
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)

    @staticmethod
    def _runtime_info() -> tuple[int, int, int]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        return rank, world_size, local_rank

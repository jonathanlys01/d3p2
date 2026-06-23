"""Configuration for the experimental distributed MAP-DPP sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistributedMAPConfig:
    """Shape and runtime options for projected batched greedy MAP-DPP."""

    sequence_length: int = 1024
    embedding_dim: int = 768
    projected_dim: int = 128
    selections: int = 32
    local_trajectories: int = 64
    block_n: int = 256
    seed: int = 0
    projection_seed: int = 17
    epsilon: float = 1e-10
    embedding_dtype: torch.dtype = torch.bfloat16
    projection_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    device: str = "auto"
    require_triton: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.projected_dim <= 0:
            raise ValueError("projected_dim must be positive")
        if self.selections <= 0:
            raise ValueError("selections must be positive")
        if self.selections > self.sequence_length:
            raise ValueError("selections cannot exceed sequence_length")
        if self.local_trajectories <= 0:
            raise ValueError("local_trajectories must be positive")
        if self.block_n <= 0:
            raise ValueError("block_n must be positive")
        if self.sequence_length % self.block_n != 0:
            raise ValueError("sequence_length must be divisible by block_n for the Triton kernel")
        if self.projected_dim != 128:
            raise ValueError("the experimental Triton kernel currently requires projected_dim=128")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    @property
    def N(self) -> int:  # noqa: N802
        return self.sequence_length

    @property
    def D(self) -> int:  # noqa: N802
        return self.embedding_dim

    @property
    def d(self) -> int:
        return self.projected_dim

    @property
    def L(self) -> int:  # noqa: N802
        return self.selections

    @property
    def S(self) -> int:  # noqa: N802
        return self.local_trajectories

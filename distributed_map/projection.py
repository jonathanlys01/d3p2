"""Johnson-Lindenstrauss projection helpers."""

from __future__ import annotations

import math

import torch

from distributed_map.config import DistributedMAPConfig


def make_projection_matrix(
    config: DistributedMAPConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic JL matrix with the same seed on every rank."""

    target_device = torch.device(device)
    generator_device = "cuda" if target_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.projection_seed)

    scale = 1.0 / math.sqrt(config.projected_dim)
    matrix = torch.randn(
        (config.embedding_dim, config.projected_dim),
        generator=generator,
        device=target_device,
        dtype=torch.float32,
    )
    return (matrix * scale).to(dtype=dtype)


def project_embeddings(
    embeddings: torch.Tensor,
    config: DistributedMAPConfig,
) -> torch.Tensor:
    """Project embeddings to the SRAM-friendly low-dimensional space."""

    if embeddings.shape != (config.sequence_length, config.embedding_dim):
        raise ValueError(
            "embeddings must have shape "
            f"({config.sequence_length}, {config.embedding_dim}), got {tuple(embeddings.shape)}",
        )

    projection = make_projection_matrix(
        config,
        device=embeddings.device,
        dtype=config.projection_dtype,
    )
    projected = embeddings.to(dtype=config.projection_dtype) @ projection
    return projected.to(dtype=config.projection_dtype).contiguous()

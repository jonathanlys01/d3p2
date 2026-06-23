"""Dummy embedding generation for the experimental sampler."""

from __future__ import annotations

import torch

from distributed_map.config import DistributedMAPConfig


def make_dummy_embeddings(
    config: DistributedMAPConfig,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate deterministic synthetic embeddings without touching real data."""

    target_device = torch.device("cpu" if device is None else device)
    generator_device = "cuda" if target_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed)

    embeddings = torch.randn(
        (config.sequence_length, config.embedding_dim),
        generator=generator,
        device=target_device,
        dtype=torch.float32,
    )
    return embeddings.to(dtype=config.embedding_dtype)

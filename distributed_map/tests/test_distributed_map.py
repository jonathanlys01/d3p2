from __future__ import annotations

import pytest
import torch

from distributed_map.config import DistributedMAPConfig
from distributed_map.dummy_data import make_dummy_embeddings
from distributed_map.kernels import HAS_TRITON, partition_initial_items
from distributed_map.projection import project_embeddings
from distributed_map.sampler import DistributedMAPSampler


def _small_config(**kwargs) -> DistributedMAPConfig:
    defaults = {
        "sequence_length": 8,
        "embedding_dim": 4,
        "projected_dim": 128,
        "selections": 3,
        "local_trajectories": 2,
        "block_n": 2,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return DistributedMAPConfig(**defaults)


def test_config_defaults_match_spec() -> None:
    config = DistributedMAPConfig()

    assert config.sequence_length == 1024
    assert config.embedding_dim == 768
    assert config.projected_dim == 128
    assert config.block_n == 256
    assert config.embedding_dtype is torch.bfloat16
    assert config.state_dtype is torch.float32


def test_dummy_embeddings_are_deterministic_and_bfloat16() -> None:
    config = _small_config(seed=123)

    first = make_dummy_embeddings(config)
    second = make_dummy_embeddings(config)

    assert first.shape == (8, 4)
    assert first.dtype is torch.bfloat16
    torch.testing.assert_close(first, second)


def test_projection_is_deterministic_for_equal_seed() -> None:
    config = _small_config(projection_seed=999)
    embeddings = make_dummy_embeddings(config)

    first = project_embeddings(embeddings, config)
    second = project_embeddings(embeddings, config)

    assert first.shape == (8, 128)
    assert first.dtype is torch.bfloat16
    torch.testing.assert_close(first, second)


def test_partition_initialization_is_partition_local_not_global() -> None:
    config = _small_config(local_trajectories=2)
    x_tilde = torch.zeros((8, 128), dtype=torch.bfloat16)
    x_tilde[1, 0] = 2.0
    x_tilde[2, 0] = 3.0
    x_tilde[5, 0] = 4.0
    x_tilde[7, 0] = 99.0

    rank0 = partition_initial_items(x_tilde, config, rank=0, world_size=2)
    rank1 = partition_initial_items(x_tilde, config, rank=1, world_size=2)

    assert rank0.tolist() == [1, 2]
    assert rank1.tolist() == [5, 7]


def test_cpu_sampler_returns_finite_unique_sequence_without_n2_kernel() -> None:
    config = _small_config(seed=7, selections=4, local_trajectories=2)
    sampler = DistributedMAPSampler(config)

    result = sampler.sample()

    assert result.selected.shape == (4,)
    assert torch.unique(result.selected).numel() == 4
    assert torch.isfinite(result.score)
    assert result.x_tilde.shape == (8, 128)
    assert result.x_tilde.dtype is torch.bfloat16
    assert result.local_kernel.cis.shape == (2, 4, 8)
    assert result.local_kernel.di2s.shape == (2, 8)
    assert result.local_kernel.cis.dtype is torch.float32
    assert result.used_triton is False


def test_epsilon_clamp_keeps_log_pivots_finite_for_duplicate_vectors() -> None:
    config = _small_config(selections=3, local_trajectories=1, epsilon=1e-10)
    x_tilde = torch.ones((8, 128), dtype=torch.bfloat16)
    sampler = DistributedMAPSampler(config)

    result = sampler.sample_projected(x_tilde)

    assert torch.isfinite(result.local_kernel.log_pivots).all()
    assert torch.isfinite(result.score)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA/Triton unavailable")
def test_cuda_triton_smoke_defaults() -> None:
    config = DistributedMAPConfig(selections=4, local_trajectories=2, require_triton=True)
    sampler = DistributedMAPSampler(config)

    result = sampler.sample()

    assert result.used_triton is True
    assert result.x_tilde.dtype is torch.bfloat16
    assert result.local_kernel.cis.dtype is torch.float32
    assert result.local_kernel.di2s.dtype is torch.float32
    assert result.selected.shape == (4,)

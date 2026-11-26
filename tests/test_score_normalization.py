"""Quality normalizations of the DPP kernel (A2 ablations), and the beta = 0 / Greedy Beam Search equivalence."""

import pytest
import torch
import torch.nn.functional as F

from d5p4.config import Cache, Config
from d5p4.resume_db import semantic_config_dict
from d5p4.subsample.base import UNIFORM_QUALITY, _compute_scores, normalize_scores
from d5p4.subsample.beam import GreedyBeamSearch
from d5p4.subsample.greedy_map import GreedyMAP


N_GROUPS, GROUP_SIZE, BLOCK, VOCAB, HIDDEN = 4, 4, 6, 11, 8


def _config(**overrides) -> Config:
    base = {
        "disable_sys_args": True,
        "model": "llada",
        "standalone_job": True,
        "method": "greedy_map",
        "transversal": True,
        "n_groups": N_GROUPS,
        "group_size": GROUP_SIZE,
        "_kernel_type": "cosine",
        "_kernel_method": "additive",
        "_w_interaction": 20.0,
    }
    return Config(**{**base, **overrides})


def _cache(seed: int = 0) -> Cache:
    generator = torch.Generator().manual_seed(seed)
    batch = N_GROUPS * GROUP_SIZE
    logits = 3 * torch.randn(batch, BLOCK, VOCAB, generator=generator)
    return Cache(
        x=torch.zeros(batch, BLOCK, dtype=torch.long),
        log_p_x0=F.log_softmax(logits, dim=-1),
        embeddings=torch.randn(batch, BLOCK, HIDDEN, generator=generator),
    )


def _raw_scores(cache: Cache) -> torch.Tensor:
    """Mean negative token entropy, before any normalization."""
    assert cache.log_p_x0 is not None
    log_p = cache.log_p_x0.float()
    return (log_p.exp() * log_p).sum(-1).mean(-1)


def test_minmax_is_the_unchanged_default():
    cache = _cache()
    raw = _raw_scores(cache)
    expected = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    torch.testing.assert_close(_compute_scores(cache), expected)
    torch.testing.assert_close(_compute_scores(cache, normalization="minmax"), expected)
    assert _compute_scores(cache).min().item() == 0.0


def test_uniform_quality_is_constant():
    scores = normalize_scores(torch.tensor([-3.0, -0.5, -1.0]), "uniform")
    assert scores.tolist() == [UNIFORM_QUALITY] * 3


def test_exp_quality_does_not_depend_on_the_pool():
    raw = _raw_scores(_cache())
    assert (raw <= 0).all()
    exp = _compute_scores(_cache(), normalization="exp")

    torch.testing.assert_close(exp, raw.exp())
    assert ((exp > 0) & (exp <= 1)).all()
    # A candidate's quality is the same whatever the rest of the pool is, unlike min-max.
    torch.testing.assert_close(normalize_scores(raw[:3], "exp"), exp[:3])
    assert not torch.allclose(normalize_scores(raw[:3], "minmax"), normalize_scores(raw, "minmax")[:3])


def test_zscore_quality_is_bounded_and_invariant_to_affine_changes():
    raw = _raw_scores(_cache())
    z = normalize_scores(raw, "zscore")

    assert ((z > 0) & (z < 1)).all()
    torch.testing.assert_close(normalize_scores(3 * raw - 7, "zscore"), z)
    assert torch.argsort(z).tolist() == torch.argsort(raw).tolist()
    assert normalize_scores(torch.full((5,), -2.0), "zscore").tolist() == [0.5] * 5


def test_kernel_diagonal_carries_the_normalized_quality():
    cache = _cache()
    beta = 20.0
    assert cache.embeddings is not None
    flat = F.normalize(cache.embeddings.reshape(cache.embeddings.size(0), -1), dim=-1)
    similarity = flat @ flat.T
    for normalization in ("minmax", "uniform", "exp", "zscore"):
        kernel = GreedyMAP(_config(_score_normalization=normalization)).compute_kernel(cache)
        assert kernel is not None
        quality = normalize_scores(_raw_scores(cache), normalization)
        torch.testing.assert_close(kernel, beta * similarity + torch.diag(quality))


def test_exp_normalization_needs_a_log_scale_score():
    with pytest.raises(AssertionError, match="exp"):
        _config(_score_normalization="exp", _score_method="self-certainty")
    with pytest.raises(AssertionError, match="Unknown _score_normalization"):
        _config(_score_normalization="rank")


def test_default_normalization_keeps_resume_hashes():
    assert "_score_normalization" not in semantic_config_dict(_config())
    assert semantic_config_dict(_config(_score_normalization="exp"))["_score_normalization"] == "exp"


@pytest.mark.parametrize("seed", range(5))
def test_beta_zero_is_transversal_greedy_beam_search(seed):
    """With beta = 0 the kernel is diag(Q): greedy MAP under the partition picks each group's best candidate."""
    cache = _cache(seed)
    d5p4 = GreedyMAP(_config(_w_interaction=0.0)).subsample(cache)
    gbs = GreedyBeamSearch(_config(method="greedy_beam", _temperature=0.0)).subsample(cache)

    assert d5p4 is not None and gbs is not None
    assert sorted(d5p4.tolist()) == sorted(gbs.tolist())


def test_no_partition_may_keep_two_children_of_one_parent():
    """D5P3: without the partition, selection is over the whole pool; the result stays a valid set of k items."""
    cache = _cache()
    # Make group 0's children the best and mutually most distinct candidates.
    assert cache.embeddings is not None and cache.log_p_x0 is not None
    cache.embeddings[:GROUP_SIZE] = 10 * torch.eye(GROUP_SIZE, BLOCK * HIDDEN).reshape(GROUP_SIZE, BLOCK, HIDDEN)
    confident = F.log_softmax(torch.full((VOCAB,), -20.0).index_fill(0, torch.tensor([0]), 20.0), dim=-1)
    cache.log_p_x0[:GROUP_SIZE] = confident
    selected = GreedyMAP(_config(transversal=False, _w_interaction=1.0)).subsample(cache)

    assert selected is not None
    assert len(set(selected.tolist())) == N_GROUPS
    assert len({index // GROUP_SIZE for index in selected.tolist()}) < N_GROUPS

from types import MethodType

import torch

from d5p4.config import Cache, Config
from d5p4.subsample.beam import GreedyBeamSearch


def _selector_with_scores(*, transversal: bool, scores: torch.Tensor) -> GreedyBeamSearch:
    config = Config(
        disable_sys_args=True,
        method="greedy_beam",
        transversal=transversal,
        n_groups=2,
        group_size=4,
        _temperature=0.0,
    )
    selector = GreedyBeamSearch(config)
    selector.distributed_mul = 2

    def compute_scores(self, cache: Cache) -> torch.Tensor:  # noqa: ARG001
        return scores

    selector.compute_scores = MethodType(compute_scores, selector)
    return selector


def test_greedy_beam_transversal_uses_global_group_count_after_gather() -> None:
    scores = torch.arange(16, dtype=torch.float32)
    selector = _selector_with_scores(transversal=True, scores=scores)

    selected = selector._transversal(Cache())

    assert selected is not None
    assert selected.tolist() == [3, 7, 11, 15]


def test_greedy_beam_non_transversal_uses_global_selection_count_after_gather() -> None:
    scores = torch.arange(16, dtype=torch.float32)
    selector = _selector_with_scores(transversal=False, scores=scores)

    selected = selector._non_transversal(Cache())

    assert selected is not None
    assert selected.numel() == 4
    assert selected.tolist() == [15, 14, 13, 12]

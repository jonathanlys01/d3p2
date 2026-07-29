"""Selector placeholder for the dedicated left-to-right beam decoder."""

import torch

from d5p4.config import Cache
from d5p4.subsample.base import BaseSelector


class LTRBeamSelection(BaseSelector):
    """Carry runtime/distributed state for classic beam; selection happens in the decoder."""

    needs_embeddings = False

    @staticmethod
    def _unused(cache: Cache) -> torch.Tensor | None:  # noqa: ARG004
        raise RuntimeError("method=ltr_beam is handled by llada_decoder=classic_beam")

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        return self._unused(cache)

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        return self._unused(cache)

import torch

from config import Cache, Config
from subsample.base import BaseSelector


class BaselineSelection(BaseSelector):
    """Baseline sampling"""

    def __init__(self, config: Config):
        super().__init__(config)
        assert config.group_size == 1

    def _sample(self):
        """Independent sampling."""
        return torch.arange(self.config.n_groups)

    def _transversal(self, cache: Cache):  # noqa: ARG002
        return self._sample()

    def _non_transversal(self, cache: Cache):  # noqa: ARG002
        return self._sample()

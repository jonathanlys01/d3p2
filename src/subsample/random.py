import numpy as np
import torch

from config import Cache
from subsample.common import BaseSubsetSelector


class RandomSelection(BaseSubsetSelector):
    """Random Subset Selector"""

    def _transversal(self, cache: Cache):  # noqa: ARG002
        if self.distributed_utils and self.distributed_utils.rank != 0:
            return None
        gr_size = self.config.group_size
        n_groups = self.config.n_groups * self.distributed_mul
        indices = torch.randint(0, gr_size, size=(n_groups,)) + torch.arange(n_groups) * gr_size
        return indices.to(self.device)

    def _non_transversal(self, cache: Cache):  # noqa: ARG002
        if self.distributed_utils and self.distributed_utils.rank != 0:
            return None
        B = self.config.n_groups * self.config.group_size * self.distributed_mul
        indices = np.random.choice(B, self.config.n_groups * self.distributed_mul, replace=False)
        return torch.from_numpy(indices).to(self.device)

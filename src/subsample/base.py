from abc import ABC

import torch

from config import Cache, Config
from utils import DistributedUtils


class BaseSubsetSelector(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda"

        self.distributed_utils = DistributedUtils(config) if DistributedUtils.is_distributed() else None
        self.distributed_mul = self.distributed_utils.world_size if self.distributed_utils else 1

    def subsample(self, cache: Cache):
        if self.config.transversal:
            return self._transversal(cache)
        else:
            return self._non_transversal(cache)

    def _transversal(self, cache: Cache) -> torch.Tensor:
        raise NotImplementedError

    def _non_transversal(self, cache: Cache) -> torch.Tensor:
        raise NotImplementedError


# General subsample utils


def compute_entropy(cache: Cache) -> torch.Tensor:
    pass  # TODO: implement


# DPP-specific utils


def compute_kernel(cache: Cache, config: Config) -> torch.Tensor:
    pass  # TODO: implement

# perplexity
# average cosine

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from transformers import AutoModel


class Metric(ABC):
    def __call__(self, embeddings: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        """
        Compute the metric on the given embeddings (shape: [num_samples, embedding_dim]).
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        return self.forward(embeddings, *args, **kwargs)

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, *args, **kwargs) -> None:
        """TODO:write docstring"""
        pass

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        pass


class Perplexity(Metric):
    def __init__(self, model_id: str):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_id)


class AverageCosine(Metric):
    def forward(self, embeddings: torch.Tensor, *args, **kwargs) -> None:
        pass

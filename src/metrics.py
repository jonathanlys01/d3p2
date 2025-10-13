# perplexity
# average cosine

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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
        pass

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        pass


class Perplexity(Metric):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, cache_dir="./.cache")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./.cache")

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        self.loss = None

    def forward(self, texts: list[str]) -> None:
        inputs = self.tokenizer(texts, return_tensor="pt")

        with torch.no_grad():
            logits: torch.Tensor = self.model(**inputs, return_dict=True).xogits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            self.loss = loss.view(shift_labels.size())

    def compute(self) -> float:
        ppl = torch.exp(self.loss.mean()).item()
        return ppl

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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        self.loss = None

        self.lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.wte.weight  # tie weights

    def forward(self, texts: list[str]) -> None:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            last_hidden_states: torch.Tensor = self.model(**inputs, return_dict=True).last_hidden_state
            logits = self.lm_head(last_hidden_states)
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


if __name__ == "__main__":
    # Example usage
    texts = [
        "This is a test sentence.",
        "Another example sentence for computing perplexity.",
    ]
    ppl_metric = Perplexity("gpt2")
    ppl_metric.forward(texts)
    perplexity = ppl_metric.compute()
    print(f"Perplexity: {perplexity}")

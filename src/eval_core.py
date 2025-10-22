# perplexity
# average cosine


import json
from dataclasses import asdict
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import CACHE_DIR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Perplexity(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        self.loss = None

        self.lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.wte.weight  # tie weights

    def _forward(self, texts: list[str]) -> None:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        self.model.to(device)

        with torch.no_grad():
            last_hidden_states: torch.Tensor = self.model(**inputs, return_dict=True).last_hidden_state
            logits = self.lm_head(last_hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss = loss.view(shift_labels.size())

        ppl = torch.exp(loss.mean()).item()
        return ppl

    def forward(self, texts: list[str], batch_size: int = 0) -> float:
        """
        Compute perplexity for a list of texts, optionally in batches.
        """
        if batch_size == 0:
            return self._forward(texts)

        total_loss = 0.0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            batch_loss = self._forward(batch_texts)
            total_loss += batch_loss

        return total_loss / n_batches


class AverageCosineSimilarity(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)

    def _forward(self, texts: list[str]) -> float:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        self.model.to(device)

        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = embeddings[:, 0, :]  # n_samples x B, D
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            x = F.normalize(x, p=2, dim=-1)

            S = x @ x.t()  # n_samples x n_samples

            S = S - torch.eye(len(texts), device=S.device)  # remove self-similarity

            avg_cos_sim = S.mean().item()

        return avg_cos_sim


# TODO: finish
def eval_bert(text_samples: list[str], cfg) -> None:
    MODEL_ID = "bert-base-uncased"
    bert = AutoModel.from_pretrained(MODEL_ID, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cfg.cache_dir)

    inputs = tokenizer(text_samples, return_tensors="pt", padding=True, truncation=True)
    embeddings = bert(**inputs, return_dict=True).last_hidden_state
    embeddings = embeddings[:, 0, :]  # n_samples x B, D

    # reshape instead of view to avoid issues with non-contiguous tensors
    x = embeddings.reshape(cfg.n_samples, cfg.batch_size, -1)  # n_samples x batch_size x D
    x = F.normalize(x, p=2, dim=-1)

    S = torch.einsum("n b d, n B d -> n b B", x, x)  # n_samples x batch_size x batch_size

    S = S - torch.eye(cfg.batch_size, device=S.device).unsqueeze(0)  # remove self-similarity

    avg_cos_sim = S.mean().item()
    print(f"Average Cosine Similarity: {avg_cos_sim}")

    # compute perplexity using a pretrained language model
    ppl_metric = Perplexity("gpt2")
    ppl_metric.forward(text_samples)
    perplexity = ppl_metric.compute()
    print(f"Average Perplexity: {perplexity}")

    results = {
        "text_samples": text_samples,
        "average_cosine_similarity": avg_cos_sim,
        "perplexity": perplexity,
        "config": asdict(cfg),
    }

    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/exp-{name}.json", "w") as f:
        json.dump(results, f, indent=4)

    if cfg.interactive:
        print(results)


def eval_all(texts: list[str], batch_size: int = 0) -> float:
    ppl_metric = Perplexity("gpt2")
    perplexity = ppl_metric(texts, batch_size=batch_size)
    return {"perplexity": perplexity}


if __name__ == "__main__":
    # Example usage
    texts = [
        "This is a test sentence.",
        "Another example sentence for computing perplexity.",
    ]
    ppl_metric = Perplexity("gpt2")
    perplexity = ppl_metric(texts, batch_size=1)
    print(f"Perplexity: {perplexity}")

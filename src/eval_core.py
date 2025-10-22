# perplexity
# average cosine


import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CACHE_DIR


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

        with torch.inference_mode():
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

    def forward(self, texts: list[list[str]], batch_size: int = 0) -> float:
        """
        Compute perplexity for a list of texts, optionally in batches.
        """

        # flatten because independent evaluation
        texts = [text for sublist in texts for text in sublist]

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

        with torch.inference_mode():
            embeddings: torch.Tensor = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = embeddings[:, 0, :]  # n_samples x B, D
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            x = F.normalize(x, p=2, dim=-1)

            S = x @ x.t()  # n_samples x n_samples

            S = S - torch.eye(len(texts), device=S.device)  # remove self-similarity

            n = S.size(0)
            avg_cos_sim = S.sum() / max(n * (n - 1), 1)  # unbiased average

        return avg_cos_sim.item()

    def forward(self, texts: list[list[str]]) -> float:
        """
        Compute average cosine similarity for a list of texts, optionally in batches.
        """

        avg_cos_sims = []
        for group in texts:
            avg_cos_sim = self._forward(group)
            avg_cos_sims.append(avg_cos_sim)

        return sum(avg_cos_sims) / len(avg_cos_sims)


class Evaluator:
    def __init__(
        self,
        batch_size: int = 0,
        force: bool = False,
        ppl_model_id: str = "gpt2",
        cos_model_id: str = "bert-base-uncased",
    ):
        self.perplexity_model = Perplexity(ppl_model_id)
        self.cosine_model = AverageCosineSimilarity(cos_model_id)

        self.batch_size = batch_size
        self.force = force

    def evaluate(self, texts: list[list[str]]) -> dict[str, float]:
        ppl = self.perplexity_model(texts, batch_size=self.batch_size)
        avg_cos_sim = self.cosine_model(texts)

        return {
            "perplexity": ppl,
            "average_cosine_similarity": avg_cos_sim,
        }

    def eval_from_file(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)

        metrics = data.get("metrics", None)
        if not self.force and metrics is not None:
            print(f"Metrics already exist in {file_path}, skipping evaluation.")
            return

        texts = data["text_samples"]
        metrics = self.evaluate(texts)

        data["metrics"] = metrics

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate text samples.")
    parser.add_argument(
        "--folder_path",
        "-f",
        type=str,
        required=True,
        help="Path to the folder containing text samples.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="Batch size for evaluation.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if metrics exist.")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.folder_path) if f.endswith(".json")]
    evaluator = Evaluator(args.batch_size, args.force)
    pbar = tqdm(files, desc="Evaluating files")

    for file_name in pbar:
        file_path = os.path.join(args.folder_path, file_name)
        evaluator.eval_from_file(file_path)
        pbar.set_postfix({"Last evaluated": file_name})


if __name__ == "__main__":
    main()

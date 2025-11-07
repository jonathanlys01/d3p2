# perplexity
# average cosine
# TODO: mauve

import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CACHE_DIR
from utils import process_model_args


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Perplexity(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        models_args = process_model_args(model_id, cache_dir=CACHE_DIR)

        self.model = AutoModel.from_pretrained(**models_args)
        self.tokenizer = AutoTokenizer.from_pretrained(**models_args)

        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        self.loss = None

        self.lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.wte.weight  # tie weights

    def _forward(self, texts: list[str]) -> torch.Tensor:
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

        ppl = torch.exp(loss.mean(dim=1))  # perplexity per sample
        return ppl.cpu().tolist()

    def forward(self, texts: list[list[str]], batch_size: int = 0) -> tuple[float, float, float, float]:
        """
        Compute perplexity for a list of texts, optionally in batches.
        """

        # flatten because independent evaluation
        texts = [text for sublist in texts for text in sublist]

        batch_size = batch_size or len(texts)

        ppls = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            ppls.extend(self._forward(batch))

        ppls_tensor = torch.tensor(ppls)

        mean_ppl = ppls_tensor.mean().item()
        min_ppl = ppls_tensor.min().item()
        max_ppl = ppls_tensor.max().item()
        std_ppl = ppls_tensor.std().item()

        _median = ppls_tensor.median().item()
        mad_ppm = torch.mean(torch.abs(ppls_tensor - _median)).item()

        return mean_ppl, min_ppl, max_ppl, std_ppl, mad_ppm


class AverageCosineSimilarity(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR, trust_remote_code=True)

    def _forward(self, texts: list[str]) -> float:
        self.model.to(device)

        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            S = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

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

        cos_sims_tensor = torch.tensor(avg_cos_sims)
        min_cos_sim = cos_sims_tensor.min().item()
        max_cos_sim = cos_sims_tensor.max().item()
        mean_cos_sim = cos_sims_tensor.mean().item()
        std_cos_sim = cos_sims_tensor.std().item() if len(cos_sims_tensor) > 1 else -1.0

        return mean_cos_sim, min_cos_sim, max_cos_sim, std_cos_sim


class Evaluator:
    def __init__(
        self,
        batch_size: int = 0,
        force: bool = False,
        ppl_model_id: str = "gpt2",
        cos_model_id: str = "jinaai/jina-embeddings-v2-base-en",
    ):
        self.perplexity_model = Perplexity(ppl_model_id)
        self.cosine_model = AverageCosineSimilarity(cos_model_id)

        self.batch_size = batch_size
        self.force = force

    def evaluate(self, texts: list[list[str]]) -> dict[str, float]:
        ppl, min_ppl, max_ppl, std_ppl, mad_ppl = self.perplexity_model(texts, batch_size=self.batch_size)
        avg_cos_sim, min_cos_sim, max_cos_sim, std_cos_sim = self.cosine_model(texts)

        return {
            # PPL
            "perplexity": ppl,
            "min_perplexity": min_ppl,
            "max_perplexity": max_ppl,
            "std_perplexity": std_ppl,
            "mad_perplexity": mad_ppl,
            # Cosine similarity
            "cosine_similarity": avg_cos_sim,
            "std_cosine_similarity": std_cos_sim,
            "min_cosine_similarity": min_cos_sim,
            "max_cosine_similarity": max_cos_sim,
        }

    def eval_from_file(self, file_path: str) -> dict[str, float] | None:
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

        return metrics


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

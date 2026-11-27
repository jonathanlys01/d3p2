"""Text features for the open-ended (MDLM) waves, computed once per run on one GPU.

For every generated continuation, and once per task for the held-out reference continuations:

- `nll`: mean per-token negative log-likelihood under GPT-2 Large (the paper's perplexity model, same tokenization,
  clamping and tied output head as `d5p4.eval_core.Perplexity`), with `tokens` its token count;
- `entropy`: entropy (nats) of the unigram distribution of the text's GPT-2 tokens, which drops for repetitive text
  that can still have a low perplexity;
- `mauve`: the MAUVE feature of Pillutla et al. (2021), the final-layer GPT-2 Large hidden state of the last token,
  taken from the same forward pass;
- `jina`: the L2-normalized Jina v2 embedding used for self-cosine in the paper.

Outputs: `ROOT/eval/<task>/<tag>/open_features.npz` with `prompt`, `slot`, the arrays above, and a `metrics.json`
with point means for `summary.csv`; the reference set goes to `ROOT/eval/<task>/_reference/open_features.npz`.
A run whose features are newer than its result file is skipped, so a re-evaluation only featurizes new runs.

    python -m d5p4.waves.open_eval --wave b4 --root ROOT [--task mdlm_owt] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Model

from d5p4.config import Config
from d5p4.data.owt_heldout import load_open_chunks
from d5p4.eval_core import AverageCosineSimilarity
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.utils import get_tokenizer, process_model_args
from d5p4.waves import get_wave
from d5p4.waves.fronts import _result_file
from d5p4.waves.spec import Run


FEATURES = "open_features.npz"
REFERENCE_DIR = "_reference"
NLL_CLAMP = 15.0  # as d5p4.eval_core.Perplexity


def token_entropy(token_ids: list[int]) -> float:
    """Entropy (nats) of the empirical unigram distribution of one text's tokens."""
    if not token_ids:
        return 0.0
    _, counts = np.unique(np.asarray(token_ids), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


class GPT2Scorer:
    """GPT-2 Large NLL and MAUVE features from one forward pass per batch."""

    def __init__(self, model_id: str, cache_dir: str, device: str):
        assert "gpt2" in model_id.lower(), f"Open-ended evaluation expects a GPT-2 model, got {model_id}"
        args = process_model_args(model_id, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(**args)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = GPT2Model.from_pretrained(**args).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def score(self, texts: list[str], batch_size: int) -> dict[str, np.ndarray]:
        texts = [t.strip() for t in texts]
        encoded = [self.tokenizer(t, add_special_tokens=False, truncation=True)["input_ids"] for t in texts]
        order = np.argsort([-len(ids) for ids in encoded], kind="stable")
        nll = np.zeros(len(texts))
        tokens = np.zeros(len(texts), dtype=np.int64)
        entropy = np.array([token_entropy(ids) for ids in encoded])
        features = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        head = self.model.wte.weight
        for start in range(0, len(texts), batch_size):
            batch = [int(i) for i in order[start : start + batch_size]]
            lengths = [max(len(encoded[i]), 1) for i in batch]
            width = max(lengths)
            ids = torch.full((len(batch), width), self.tokenizer.eos_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                seq = encoded[i] or [self.tokenizer.eos_token_id]
                ids[row, : len(seq)] = torch.tensor(seq)
                mask[row, : len(seq)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            hidden = self.model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            logits = hidden[:, :-1] @ head.T
            token_nll = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="none",
            ).view(len(batch), -1)
            token_nll = torch.nan_to_num(token_nll, nan=NLL_CLAMP, posinf=NLL_CLAMP, neginf=0.0).clamp(max=NLL_CLAMP)
            valid = mask[:, 1:].bool()
            counts = valid.sum(dim=1)
            sums = (token_nll * valid).sum(dim=1)
            mean = torch.where(counts > 0, sums / counts.clamp(min=1), torch.full_like(sums, NLL_CLAMP))
            last = torch.tensor(lengths, device=self.device) - 1
            feats = hidden[torch.arange(len(batch), device=self.device), last]
            for row, i in enumerate(batch):
                nll[i] = float(mean[row])
                tokens[i] = int(counts[row])
                features[i] = feats[row].float().cpu().numpy()
        return {"nll": nll, "tokens": tokens, "entropy": entropy, "mauve": features}


class JinaEmbedder:
    def __init__(self, model_id: str, cache_dir: str):
        self.encoder = AverageCosineSimilarity(
            JinaBertModel.from_pretrained(**process_model_args(model_id, cache_dir=cache_dir)),
        )

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        parts = [
            self.encoder._encode(texts[i : i + batch_size]).float().cpu().numpy()
            for i in range(0, len(texts), batch_size)
        ]
        return np.concatenate(parts).astype(np.float32)


def flatten_run(data: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Texts of one result file with their prompt index and slot (position within the prompt's texts)."""
    texts, prompts, slots = [], [], []
    for index, group in zip(data["dataset_indices"], data["text_samples"], strict=True):
        for slot, text in enumerate(group):
            texts.append(text)
            prompts.append(index)
            slots.append(slot)
    return texts, np.asarray(prompts, dtype=np.int64), np.asarray(slots, dtype=np.int64)


def run_config(data: dict[str, Any]) -> Config:
    # Saved configs carry disable_sys_args themselves, so it is dropped here and set once below.
    valid = {f.name for f in fields(Config)} - {"disable_sys_args"}
    return Config(**{k: v for k, v in data["config"].items() if k in valid}, disable_sys_args=True)


def reference_texts(config: Config) -> list[str]:
    tokenizer = get_tokenizer(config, "mdlm")
    chunks = load_open_chunks(config, tokenizer)
    return [tokenizer.decode(chunks.continuation(c), skip_special_tokens=True).strip() for c in chunks.references]


def point_metrics(features: dict[str, np.ndarray]) -> dict[str, float]:
    """Means over every text of a run, for summary.csv; the paired summaries come from d5p4.waves.open_fronts."""
    return {
        "texts": int(len(features["nll"])),
        "perplexity": float(np.exp(features["nll"].mean())),
        "nll": float(features["nll"].mean()),
        "token_entropy": float(features["entropy"].mean()),
        "tokens": float(features["tokens"].mean()),
    }


def _fresh(output: Path, source: Path) -> bool:
    return output.exists() and output.stat().st_mtime >= source.stat().st_mtime


def evaluate_task(runs: list[Run], root: Path, batch_size: int, device: str) -> None:
    task = runs[0].task
    done = [(run, _result_file(str(run.results_dir(root)))) for run in runs]
    done = [(run, Path(path)) for run, path in done if path is not None]
    if not done:
        print(f"No results for {task.name}")
        return
    eval_dir = root / "eval" / task.name
    first = json.loads(done[0][1].read_text())
    config = run_config(first)
    scorer = GPT2Scorer(config.ppl_model_id, config.cache_dir, device)
    embedder = JinaEmbedder(config.cos_model_id, config.cache_dir)

    reference = eval_dir / REFERENCE_DIR / FEATURES
    if not reference.exists():
        texts = reference_texts(config)
        print(f"{task.name}: featurizing {len(texts)} reference continuations")
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference, **scorer.score(texts, batch_size))
        _write_metrics(reference.parent, REFERENCE_DIR, dict(np.load(reference)))

    for run, path in done:
        output = eval_dir / run.tag / FEATURES
        if _fresh(output, path):
            continue
        data = json.loads(path.read_text())
        texts, prompts, slots = flatten_run(data)
        print(f"{task.name}/{run.tag}: featurizing {len(texts)} texts from {path.name}")
        features = {**scorer.score(texts, batch_size), "jina": embedder.embed(texts), "prompt": prompts, "slot": slots}
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **features)
        _write_metrics(output.parent, f"{run.tag}/{path.name}", features)


def _write_metrics(directory: Path, source: str, features: dict[str, np.ndarray]) -> None:
    payload = {"source_relative_path": source, "evaluation_kind": "open", "metrics": point_metrics(features)}
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", default=None, help="Only this task (default: every open task of the wave).")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [run for run in get_wave(args.wave) if run.task.family == "open"]
    for name in dict.fromkeys(run.task.name for run in runs):
        if args.task is None or args.task == name:
            evaluate_task([run for run in runs if run.task.name == name], args.root.resolve(), args.batch_size, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

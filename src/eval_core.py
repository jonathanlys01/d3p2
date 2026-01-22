"""
Core evaluation module for calculating text generation metrics.
Includes implementations for Perplexity, MAUVE, SacreBLEU, and
Cosine Similarity (Jina BERT).
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import ot
import sacrebleu
import torch
import torch.nn.functional as F
from scipy.stats._continuous_distns import t
from transformers import AutoModel, AutoTokenizer, GPT2Model, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

import mauve
from config import CACHE_DIR
from jina_ref.modeling_bert import JinaBertModel
from utils import print as u_print
from utils import process_model_args, tqdm


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_statistics(values: list[float], prefix: str) -> dict[str, float]:
    """
    Compute comprehensive statistics for a list of values.
    Returns dictionary with keys formatted as {prefix}_{stat}.
    The main value (mean) is also returned as {prefix} for backward compatibility.
    """
    # Filter valid values (finite numbers)
    valid_values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]

    stats = {}
    n = len(valid_values)

    if n == 0:
        # Return NaNs for empty/invalid input
        keys = ["mean", "median", "min", "max", "std", "mad", "stderr", "ci95"]
        for k in keys:
            stats[f"{prefix}_{k}"] = float("nan")
        stats[prefix] = float("nan")
        stats[f"{prefix}_count"] = 0.0
        return stats

    data = np.array(valid_values)
    mean_val = np.mean(data).item()
    median_val = np.median(data).item()
    min_val = np.min(data).item()
    max_val = np.max(data).item()
    std_val = np.std(data, ddof=1).item() if n > 1 else 0.0
    mad_val = np.mean(np.abs(data - median_val)).item()

    stderr_val = std_val / np.sqrt(n) if n > 0 else 0.0

    # Determine critical value for 95% CI using student-t distribution
    if 1 < n < 30:
        critical_value = float(t.ppf(0.975, df=n - 1))
    elif n >= 30:
        critical_value = 1.96
    else:
        # For n=1, stderr is 0.0, so CI is 0.0
        critical_value = 0.0

    ci95_val = critical_value * stderr_val  # 95% Confidence Interval

    stats[prefix] = mean_val
    stats[f"{prefix}_mean"] = mean_val
    stats[f"{prefix}_median"] = median_val
    stats[f"{prefix}_min"] = min_val
    stats[f"{prefix}_max"] = max_val
    stats[f"{prefix}_std"] = std_val
    stats[f"{prefix}_mad"] = mad_val
    stats[f"{prefix}_stderr"] = stderr_val
    stats[f"{prefix}_ci95"] = ci95_val
    stats[f"{prefix}_count"] = float(n)

    return stats


def _format_num(x: float, sig_figs: int = 4) -> str:
    """Format a number with specified significant figures."""
    if x == 0:
        return "0"
    if np.isnan(x):
        return "NaN"
    return f"{x:.{sig_figs}g}"


def _format_summary_value(mean: float, ci95: float, sig_figs: int = 4) -> str:
    """Format a mean and symmetric CI value."""
    return f"{_format_num(mean, sig_figs)} pm {_format_num(ci95, sig_figs)}"


def _format_asymmetric_ci(mean: float, lower: float, upper: float, sig_figs: int = 4) -> str:
    """Format a mean and asymmetric CI bounds."""
    return f"{_format_num(mean, sig_figs)} [{_format_num(lower, sig_figs)}, {_format_num(upper, sig_figs)}]"


class Perplexity(torch.nn.Module):
    def __init__(self, model: AutoModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        self.loss = None

        if isinstance(self.model, GPT2Model):
            self.lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
            self.lm_head.weight = self.model.wte.weight  # tie weights
        elif isinstance(self.model, LlamaForCausalLM):
            self.lm_head = self.model.lm_head  # reference model's existing lm_head
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _forward(self, texts: list[str]) -> list[float] | None:
        """Compute per-sample mean NLL (loss) values. Statistics should be computed in this space."""
        texts = [t.strip() for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        # Skip forward pass if inputs are empty (0 tokens) to avoid reshape errors
        if inputs["input_ids"].numel() == 0:
            return None

        self.model.to(device)

        with torch.inference_mode():
            if isinstance(self.model, LlamaForCausalLM):
                outputs: CausalLMOutputWithPast = self.model(**inputs, return_dict=True, output_hidden_states=True)
                assert outputs.hidden_states is not None
                last_hidden_states = outputs.hidden_states[-1]
            else:
                last_hidden_states: torch.Tensor = self.model(**inputs, return_dict=True).last_hidden_state
            logits = self.lm_head(last_hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            attention_mask = inputs["attention_mask"][..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = loss.view(shift_labels.size())

        loss = loss.clamp(max=15.0)
        loss = loss * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_loss = loss.sum(dim=1) / token_counts
        mean_loss = torch.nan_to_num(mean_loss, nan=15.0, posinf=15.0, neginf=0.0)

        return mean_loss.cpu().tolist()

    def forward(self, texts: list[list[str]], batch_size: int = 0) -> dict[str, float]:
        """
        Compute perplexity statistics. Statistics are computed in NLL (loss) space,
        then transformed to perplexity space via exp().
        """
        flattened_texts = [text for sublist in texts for text in sublist]
        batch_size = batch_size or len(flattened_texts)

        # Collect per-sample NLL values
        nlls = []
        for start in range(0, len(flattened_texts), batch_size):
            batch = flattened_texts[start : start + batch_size]
            result = self._forward(batch)
            if result is not None:
                nlls.extend(result)

        # Compute statistics in NLL space
        nll_stats = compute_statistics(nlls, "nll")

        # Transform to perplexity space: PPL = exp(NLL)
        ppl_stats = {
            "perplexity": np.exp(nll_stats["nll"]),
            "perplexity_mean": np.exp(nll_stats["nll_mean"]),
            "perplexity_median": np.exp(nll_stats["nll_median"]),
            "perplexity_min": np.exp(nll_stats["nll_min"]),
            "perplexity_max": np.exp(nll_stats["nll_max"]),
            "perplexity_ci95_lower": np.exp(nll_stats["nll_mean"] - nll_stats["nll_ci95"]),
            "perplexity_ci95_upper": np.exp(nll_stats["nll_mean"] + nll_stats["nll_ci95"]),
            "perplexity_count": nll_stats["nll_count"],
        }

        return ppl_stats


class AverageCosineSimilarity(torch.nn.Module):
    def __init__(self, model: JinaBertModel):
        super().__init__()
        self.model = model

    def _forward(self, texts: list[str]) -> float:
        self.model.to(device)

        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)  # type: ignore
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            S = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

            S = S - torch.eye(len(texts), device=S.device)  # remove self-similarity

            n = S.size(0)
            if n <= 1:
                return 0.0
            avg_cos_sim = S.sum() / (n * (n - 1))  # unbiased average

        return avg_cos_sim.item()

    def forward(self, texts: list[list[str]]) -> dict[str, float]:
        """
        Compute average cosine similarity statistics for a list of texts of groups.
        """

        avg_cos_sims = []
        for group in texts:
            avg_cos_sim = self._forward(group)
            avg_cos_sims.append(avg_cos_sim)

        return compute_statistics(avg_cos_sims, "cosine_similarity")


class MAUVE(torch.nn.Module):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, p_text: list[str], q_text: list[str]):
        """
        Compute MAUVE score for a list of texts using the mauve package.
        """

        out = mauve.compute_mauve(
            p_text=p_text,
            q_text=q_text,
            models=(self.model, self.tokenizer),
            device_id=0 if torch.cuda.is_available() else -1,
        )

        return out


class WassersteinDistance(torch.nn.Module):
    def __init__(self, model: JinaBertModel):
        super().__init__()

        self.model = model

    def forward(
        self,
        good_references: list[str],
        bad_references: list[str],
        generations: list[str],
    ) -> tuple[float, float]:
        n_good = len(good_references)
        n_bad = len(bad_references)
        n_gen = len(generations)
        all_texts = generations + good_references + bad_references
        embeddings = self._forward(all_texts).numpy()

        gen_embeddings = embeddings[0:n_gen]
        good_embeddings = embeddings[n_gen : n_gen + n_good]
        bad_embeddings = embeddings[n_gen + n_good :]

        # Compute cost matrices
        cost_good = ot.dist(gen_embeddings, good_embeddings, metric="euclidean")
        cost_bad = ot.dist(gen_embeddings, bad_embeddings, metric="euclidean")

        # Uniform distributions
        p_gen = np.ones((n_gen,)) / n_gen
        p_good = np.ones((n_good,)) / n_good
        p_bad = np.ones((n_bad,)) / n_bad

        wasserstein_good: float = ot.emd2(p_gen, p_good, cost_good)  # type: ignore
        wasserstein_bad: float = ot.emd2(p_gen, p_bad, cost_bad)  # type: ignore

        return wasserstein_good, wasserstein_bad

    def _forward(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)  # type: ignore
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            x = F.normalize(x, p=2, dim=-1)
        return x.cpu()


class StringMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = prediction.lower().split()
        ground_truth_tokens = ground_truth.lower().split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_distinct_n(self, texts: list[str], n: int = 2) -> float:
        """
        Calculate the ratio of unique n-grams to total n-grams across a list of strings.
        Uses simple whitespace tokenization.
        """
        if not texts:
            return 0.0

        all_ngrams = []
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < n:
                continue
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        return unique_ngrams / total_ngrams

    def compute_self_bleu(self, texts: list[str]) -> float:
        """
        Calculate Self-BLEU: for each string, compute its BLEU score using all other strings as references.
        Returns the average score across all strings.
        """
        if len(texts) <= 1:
            return 0.0

        bleu_scores = []
        for i, hypothesis in enumerate(texts):
            # Use all other texts as references
            references = [texts[j] for j in range(len(texts)) if j != i]
            if not references:
                continue

            # Calculate sentence BLEU with all other texts as references
            bleu = sacrebleu.sentence_bleu(hypothesis, references)
            bleu_scores.append(bleu.score)

        if not bleu_scores:
            return 0.0

        return sum(bleu_scores) / len(bleu_scores)

    def forward(self, predictions: list[list[str]], references: list[list[str]] | None = None) -> dict[str, float]:  # noqa: C901
        """
        Compute F1, BLEU, Distinct-2, and Self-BLEU scores.
        predictions is a list of lists of strings (multiple generations per question).
        references is a list of lists of strings (multiple possible answers per question).
        If references is None or empty, only diversity metrics are computed.
        """
        all_metrics = {}

        # Compute reference-based metrics (F1 and BLEU) if references are provided
        if references and any(refs for refs in references):
            flattened_predictions = []
            flattened_references = []
            for preds, refs in zip(predictions, references):
                for pred in preds:
                    flattened_predictions.append(pred)
                    flattened_references.append(refs)

            f1_scores = []
            for pred, refs in zip(flattened_predictions, flattened_references):
                best_f1 = max([self._compute_f1(pred, ref) for ref in refs]) if refs else 0.0
                f1_scores.append(best_f1)

            # Compute stats for F1
            all_metrics.update(compute_statistics(f1_scores, "f1"))

            bleu_score = 0.0
            if len(flattened_references) > 0:
                max_refs = max(len(refs) for refs in flattened_references)
                formatted_refs = []
                for i in range(max_refs):
                    ref_list = []
                    for refs in flattened_references:
                        if i < len(refs):
                            ref_list.append(refs[i])
                        else:
                            ref_list.append(refs[0])  # duplicate first if fewer refs
                    formatted_refs.append(ref_list)

                bleu = sacrebleu.corpus_bleu(flattened_predictions, formatted_refs)
                bleu_score = bleu.score

            all_metrics["bleu"] = bleu_score

        # Compute Distinct-2 and Self-BLEU per group
        distinct_2_scores = []
        self_bleu_scores = []
        for group in predictions:
            if group:  # Handle empty groups gracefully
                distinct_2_scores.append(self.compute_distinct_n(group, n=2))
                self_bleu_scores.append(self.compute_self_bleu(group))

        all_metrics.update(compute_statistics(distinct_2_scores, "distinct_2"))
        all_metrics.update(compute_statistics(self_bleu_scores, "self_bleu"))

        return all_metrics


class Evaluator:
    def __init__(
        self,
        batch_size: int = 0,
        force: bool = False,
        ppl_model_id: str = "gpt2",
        cos_model_id: str = "jinaai/jina-embeddings-v2-base-en",
    ):
        ppl_models_args = process_model_args(ppl_model_id, cache_dir=CACHE_DIR)
        if "llama" in ppl_model_id:
            ppl_model = LlamaForCausalLM.from_pretrained(**ppl_models_args)
        else:
            ppl_model = AutoModel.from_pretrained(**ppl_models_args)

        ppl_tokenizer = AutoTokenizer.from_pretrained(**ppl_models_args)
        self.perplexity_model = Perplexity(ppl_model, ppl_tokenizer)
        self.mauve_model = MAUVE(ppl_model, ppl_tokenizer)  # reuse PPL model for MAUVE (gpt2)

        cos_models_args = process_model_args(cos_model_id, cache_dir=CACHE_DIR)
        cos_model = JinaBertModel.from_pretrained(**cos_models_args)
        self.cosine_model = AverageCosineSimilarity(cos_model)
        self.wasserstein_model = WassersteinDistance(cos_model)  # reuse COS model for WD
        self.string_metrics = StringMetrics()

        self.batch_size = batch_size
        self.force = force

    def evaluate(self, texts: list[list[str]]) -> dict[str, float]:
        # Compute all metrics
        ppl_stats = self.perplexity_model(texts, batch_size=self.batch_size)
        cos_stats = self.cosine_model(texts)
        string_stats = self.string_metrics(texts, references=None)

        # Merge all metrics
        metrics = {**ppl_stats, **cos_stats, **string_stats}

        # create a summary string
        summary_parts = []

        # Define metrics to include in summary with display names
        summary_targets = [
            ("perplexity", "PPL"),
            ("cosine_similarity", "CosSim"),
            ("distinct_2", "Dist-2"),
            ("self_bleu", "S-BLEU"),
        ]

        for key, display_name in summary_targets:
            # Handle asymmetric CIs (perplexity) and symmetric CIs (other metrics)
            if key == "perplexity" and f"{key}_ci95_lower" in metrics:
                val_str = _format_asymmetric_ci(
                    metrics[key],
                    metrics[f"{key}_ci95_lower"],
                    metrics[f"{key}_ci95_upper"],
                )
                summary_parts.append(f"{display_name}: {val_str}")
            elif key in metrics and f"{key}_ci95" in metrics:
                val_str = _format_summary_value(metrics[key], metrics[f"{key}_ci95"])
                summary_parts.append(f"{display_name}: {val_str}")

        if summary_parts:
            metrics["metrics_summary"] = " | ".join(summary_parts)

        return metrics

    def compute_mauve(self, references: list[str], generations: list[str]) -> float:
        out = self.mauve_model(references, generations)
        return out.mauve

    def compute_wasserstein_distance(
        self,
        generations: list[str],
        good_references: list[str],
        bad_references: list[str],
    ) -> tuple[float, float]:
        return self.wasserstein_model(good_references, bad_references, generations)

    def compute_string_metrics(self, predictions: list[list[str]], references: list[list[str]]) -> dict[str, float]:
        return self.string_metrics(predictions, references)

    def evaluate_baseline(self, full_sequences: list[list[str]], metric: str, k: int) -> list[list[str]]:
        """
        Evaluate and select the k best sequences across different groups based on a metric.
        Initially implemented for PPL (lower is better).
        Returns the subset that minimizes NLL (lower NLL = lower PPL).
        """
        if metric.lower() != "ppl":
            raise ValueError(f"Metric {metric} not implemented for evaluate_baseline. Only 'ppl' is supported.")

        flattened_texts = [text for sublist in full_sequences for text in sublist]
        group_sizes = [len(sublist) for sublist in full_sequences]

        batch_size = self.batch_size or len(flattened_texts)
        nlls = []
        for start in range(0, len(flattened_texts), batch_size):
            batch = flattened_texts[start : start + batch_size]
            result = self.perplexity_model._forward(batch)
            if result is not None:
                nlls.extend(result)
            else:
                u_print("Skipping batch of empty texts", batch)

        # Unflatten scores to match group structure
        unflattened_nlls = []
        cursor = 0
        for size in group_sizes:
            unflattened_nlls.append(nlls[cursor : cursor + size])
            cursor += size

        # Select k best (lowest NLL) from each group
        selected_sequences = []
        for group_texts, group_nlls in zip(full_sequences, unflattened_nlls):
            indexed_nlls = sorted(enumerate(group_nlls), key=lambda x: x[1])
            top_k_indices = [idx for idx, _ in indexed_nlls[:k]]
            selected_sequences.append([group_texts[idx] for idx in top_k_indices])

        return selected_sequences

    def eval_from_file(self, file_path: str) -> dict[str, float] | None:
        with open(file_path, "r") as f:
            data = json.load(f)

        metrics = data.get("metrics", None)
        if not self.force and metrics is not None:
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

    files = [f for f in os.listdir(args.folder_path) if f.endswith(".json") and not f.startswith("temp")]
    evaluator = Evaluator(args.batch_size, args.force)
    pbar = tqdm(files, desc="Evaluating files")

    for file_name in pbar:
        file_path = os.path.join(args.folder_path, file_name)
        evaluator.eval_from_file(file_path)


if __name__ == "__main__":
    main()

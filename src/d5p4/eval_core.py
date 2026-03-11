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
from nltk.util import ngrams
from scipy.stats._continuous_distns import t
from transformers import AutoModel, AutoTokenizer, GPT2Model, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from d5p4 import mauve
from d5p4.config import CACHE_DIR
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.text_postprocessors import MathParser, universal_math_postprocess
from d5p4.utils import print as u_print
from d5p4.utils import process_model_args, tqdm


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

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to normalized embeddings."""
        self.model.to(device)
        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)  # type: ignore
            x = embeddings.reshape(len(texts), -1)  # n_samples x D
            x = F.normalize(x, p=2, dim=-1)
        return x

    def _forward(self, texts: list[str]) -> float:
        if isinstance(texts, str):
            texts = [texts]

        x = self._encode(texts)  # [n_samples, D], already normalized
        S = torch.mm(x, x.t())  # cosine similarity matrix (since x is normalized)

        S = S - torch.eye(len(texts), device=S.device)  # remove self-similarity

        n = S.size(0)
        if n <= 1:
            return 0.0
        avg_cos_sim = S.sum() / (n * (n - 1))  # unbiased average

        return avg_cos_sim.item()

    def compute_max_alignment(
        self,
        predictions: list[list[str]],
        references: list[list[str]],
    ) -> list[float]:
        """
        For each question (group), compute max cosine similarity between any prediction and any reference.
        Returns a list of max alignment scores (one per question).
        """
        max_alignments = []
        for preds, refs in zip(predictions, references):
            if not preds or not refs:
                max_alignments.append(0.0)
                continue

            # Encode all predictions and references for this question
            all_texts = preds + refs
            embeddings = self._encode(all_texts)

            pred_embs = embeddings[: len(preds)]  # [num_preds, D]
            ref_embs = embeddings[len(preds) :]  # [num_refs, D]

            # Compute cosine similarity matrix: [num_preds, num_refs]
            sim_matrix = torch.mm(pred_embs, ref_embs.t())

            # Take max across all pred-ref pairs
            max_sim = sim_matrix.max().item()
            max_alignments.append(max_sim)

        return max_alignments

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
        generations: list[str],
        good_references: list[str],
        bad_references: list[str] | None = None,
    ) -> tuple[float, float]:
        n_good = len(good_references)
        n_gen = len(generations)

        all_texts = generations + good_references
        if bad_references:
            n_bad = len(bad_references)
            all_texts += bad_references
        else:
            n_bad = 0

        embeddings = self._forward(all_texts).numpy()

        gen_embeddings = embeddings[0:n_gen]
        good_embeddings = embeddings[n_gen : n_gen + n_good]

        # Compute cost matrices
        cost_good = ot.dist(gen_embeddings, good_embeddings, metric="euclidean")

        # Uniform distributions
        p_gen = np.ones((n_gen,)) / n_gen
        p_good = np.ones((n_good,)) / n_good

        wasserstein_good: float = ot.emd2(p_gen, p_good, cost_good)  # type: ignore

        wasserstein_bad = float("nan")
        if bad_references and n_bad > 0:
            bad_embeddings = embeddings[n_gen + n_good :]
            cost_bad = ot.dist(gen_embeddings, bad_embeddings, metric="euclidean")
            p_bad = np.ones((n_bad,)) / n_bad
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
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    def compute_distinct_metrics(
        self,
        texts: list[str],
        vocab_size: int | None = None,
        references_for_vocab: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Calculate robust distinct metrics including EAD, Dist-1, Dist-2, Dist-3.
        """
        if not texts:
            return {}

        # Determine vocab size if EAD is needed
        if vocab_size is None and references_for_vocab is not None:
            vocab = set()
            for sentence in references_for_vocab:
                vocab.update(self.tokenizer.tokenize(sentence))
            vocab_size = len(vocab)

        # If vocab_size is still None, we can't compute EAD, but we can compute others.
        # But for now let's assume if it is None we just skip EAD or use a default?
        # The user code raises error if dataForVocabCal is None.
        # But we want to be robust. If no refs, maybe skip EAD?

        distinct_tokens = set()
        distinct_tokens_2grams = set()
        distinct_tokens_3grams = set()
        total_tokens = []
        total_tokens_2grams = []
        total_tokens_3grams = []

        for prediction in texts:
            tokens = self.tokenizer.tokenize(prediction)

            # NLTK ngrams yields generator, convert to list
            tokens_2grams = list(ngrams(tokens, 2, pad_left=True, left_pad_symbol="<s>"))
            tokens_3grams = list(ngrams(tokens, 3, pad_left=True, left_pad_symbol="<s>"))

            distinct_tokens.update(tokens)
            distinct_tokens_2grams.update(tokens_2grams)
            distinct_tokens_3grams.update(tokens_3grams)

            total_tokens.extend(tokens)
            total_tokens_2grams.extend(tokens_2grams)
            total_tokens_3grams.extend(tokens_3grams)

        metrics = {}
        metrics["distinct_1"] = len(distinct_tokens) / len(total_tokens) if total_tokens else 0.0
        metrics["distinct_2"] = len(distinct_tokens_2grams) / len(total_tokens_2grams) if total_tokens_2grams else 0.0
        metrics["distinct_3"] = len(distinct_tokens_3grams) / len(total_tokens_3grams) if total_tokens_3grams else 0.0

        if vocab_size is not None and len(total_tokens) > 0:
            try:
                ead = len(distinct_tokens) / (vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** len(total_tokens)))
                metrics["expectation_adjusted_distinct"] = ead
            except ZeroDivisionError:
                metrics["expectation_adjusted_distinct"] = 0.0

        return metrics

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

    def compute_diversity_batch(
        self,
        texts: list[str],
        references_for_vocab: list[str] | None = None,
        prefix: str = "batch",
    ) -> dict[str, float]:
        """Compute diversity metrics (distinct-N, EAD, self-BLEU) over a flat
        list of texts treated as a single batch, rather than averaging per-group.

        Parameters
        ----------
        texts:
            All texts to evaluate together (e.g. every generation flattened).
        references_for_vocab:
            Optional reference strings used to estimate vocabulary size for EAD.
        prefix:
            Key prefix for the returned metric dictionary (default ``"batch"``).
        """
        if not texts:
            return {}

        distinct_metrics = self.compute_distinct_metrics(
            texts,
            references_for_vocab=references_for_vocab,
        )
        self_bleu = self.compute_self_bleu(texts)

        result: dict[str, float] = {}
        for k, v in distinct_metrics.items():
            result[f"{prefix}_{k}"] = v
        result[f"{prefix}_self_bleu"] = self_bleu
        return result

    def forward(self, predictions: list[list[str]], references: list[list[str]] | None = None) -> dict[str, float]:  # noqa: C901, PLR0912, PLR0915
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

            # Compute metric@k variants: max F1 and max BLEU across k predictions per question
            k = len(predictions[0]) if predictions and predictions[0] else 0
            all_metrics["k"] = float(k)

            if k > 0:
                # F1@k: for each question, find the prediction with the best F1 against references
                f1_at_k_scores = []
                for preds, refs in zip(predictions, references):
                    if not preds or not refs:
                        continue
                    # Compute F1 for each prediction against all references, take max
                    best_f1_for_question = 0.0
                    for pred in preds:
                        f1_for_pred = max([self._compute_f1(pred, ref) for ref in refs])
                        best_f1_for_question = max(best_f1_for_question, f1_for_pred)
                    f1_at_k_scores.append(best_f1_for_question)

                all_metrics.update(compute_statistics(f1_at_k_scores, "f1_at_k"))

                # BLEU@k: for each question, find the prediction with best sentence BLEU
                bleu_at_k_scores = []
                for preds, refs in zip(predictions, references):
                    if not preds or not refs:
                        continue
                    # Compute sentence BLEU for each prediction against references, take max
                    best_bleu_for_question = 0.0
                    for pred in preds:
                        bleu_result = sacrebleu.sentence_bleu(pred, refs)
                        best_bleu_for_question = max(best_bleu_for_question, bleu_result.score)
                    bleu_at_k_scores.append(best_bleu_for_question)

                all_metrics.update(compute_statistics(bleu_at_k_scores, "bleu_at_k"))

        # Compute Distinct-N and EAD per group
        # Prepare vocab (using all references if clear, or maybe just group refs?
        # Usually EAD is about the 'language' vocab, so all refs is better)
        vocab_ref_tokens = []
        if references and any(refs for refs in references):
            for sublist in references:
                vocab_ref_tokens.extend(sublist)

        distinct_metrics_list = []
        self_bleu_scores = []
        for group in predictions:
            if group:
                # We can calculate vocab size purely from references if available
                # Passing all references as a proxy for 'language vocabulary'
                d_metrics = self.compute_distinct_metrics(
                    group,
                    references_for_vocab=vocab_ref_tokens if vocab_ref_tokens else None,
                )
                distinct_metrics_list.append(d_metrics)

                # Compute Self-BLEU for the group
                sb_score = self.compute_self_bleu(group)
                self_bleu_scores.append(sb_score)

        # Aggregate dicts
        if distinct_metrics_list:
            keys = distinct_metrics_list[0].keys()
            for key in keys:
                values = [m[key] for m in distinct_metrics_list if key in m]
                all_metrics.update(compute_statistics(values, key))
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

    def evaluate(self, texts: list[list[str]], references: list[list[str]] | None = None) -> dict[str, float]:
        # Compute all metrics
        ppl_stats = self.perplexity_model(texts, batch_size=self.batch_size)
        cos_stats = self.cosine_model(texts)
        string_stats = self.compute_string_metrics(texts, references=references)

        # Compute Wasserstein Distance if references are provided
        wd_stats = {}
        if references and any(refs for refs in references):
            wd_scores = []
            for group_gen, group_ref in zip(texts, references):
                if not group_gen or not group_ref:
                    continue
                # We only have "good" references in this context usually
                wd_good, _ = self.wasserstein_model(group_gen, group_ref, bad_references=None)
                wd_scores.append(wd_good)
            wd_stats = compute_statistics(wd_scores, "wasserstein_distance")

        # Merge all metrics
        metrics = {**ppl_stats, **cos_stats, **string_stats, **wd_stats}

        # create a summary string
        summary_parts = []

        # Define metrics to include in summary with display names
        summary_targets = [
            ("perplexity", "PPL"),
            ("cosine_similarity", "CosSim"),
            ("wasserstein_distance", "WD"),
            ("distinct_2", "Dist-2"),
            ("self_bleu", "S-BLEU"),
            ("cos_at_k", "Cos@k"),
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
        bad_references: list[str] | None = None,
    ) -> tuple[float, float]:
        return self.wasserstein_model(generations, good_references, bad_references)

    def compute_string_metrics(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
    ) -> dict[str, float]:
        metrics = self.string_metrics(predictions, references)

        # Compute cos@k: max cosine alignment between predictions and references
        if references and any(refs for refs in references):
            cos_at_k_scores = self.cosine_model.compute_max_alignment(predictions, references)
            metrics.update(compute_statistics(cos_at_k_scores, "cos_at_k"))

        return metrics

    def evaluate_baseline(  # noqa: C901
        self,
        full_sequences: list[list[str]],
        metric: str,
        k: int,
        references: list[list[str]] | None = None,
    ) -> list[list[str]]:
        """
        Evaluate and select the k best sequences across different groups based on a metric.
        Supported metrics:
        - "ppl": Lower is better.
        - "f1": Higher is better. Requires references.
        """
        flattened_texts = [text for sublist in full_sequences for text in sublist]
        group_sizes = [len(sublist) for sublist in full_sequences]

        # Unflatten helper
        def unflatten(flat_list):
            unflattened = []
            cursor = 0
            for size in group_sizes:
                unflattened.append(flat_list[cursor : cursor + size])
                cursor += size
            return unflattened

        if metric.lower() == "ppl":
            batch_size = self.batch_size or len(flattened_texts)
            nlls = []
            for start in range(0, len(flattened_texts), batch_size):
                batch = flattened_texts[start : start + batch_size]
                result = self.perplexity_model._forward(batch)
                if result is not None:
                    nlls.extend(result)
                else:
                    u_print("Skipping batch of empty texts", batch)

            unflattened_scores = unflatten(nlls)
            reverse_sort = False  # Lower is better

        elif metric.lower() == "f1":
            if references is None:
                raise ValueError("References must be provided for f1 metric.")

            # references are [group1_refs, group2_refs, ...]
            # full_sequences are [group1_cands, group2_cands, ...]
            # We compute F1 for each candidate in group i against group i refs

            unflattened_scores = []
            for group_cands, group_refs in zip(full_sequences, references):
                group_f1 = []
                for cand in group_cands:
                    # Max F1 against any reference for this question
                    best_f1 = (
                        max([self.string_metrics._compute_f1(cand, ref) for ref in group_refs]) if group_refs else 0.0
                    )
                    group_f1.append(best_f1)
                unflattened_scores.append(group_f1)

            reverse_sort = True  # Higher is better

        else:
            raise ValueError(
                f"Metric {metric} not implemented for evaluate_baseline. Only 'ppl' and 'f1' are supported.",
            )

        # Select k best from each group
        selected_sequences = []
        for group_texts, group_scores in zip(full_sequences, unflattened_scores):
            # Sort by score
            indexed_scores = sorted(enumerate(group_scores), key=lambda x: x[1], reverse=reverse_sort)
            top_k_indices = [idx for idx, _ in indexed_scores[:k]]

            # Preserve original order for selected items (optional, but cleaner)
            top_k_indices.sort()

            selected_sequences.append([group_texts[idx] for idx in top_k_indices])

        return selected_sequences

    def eval_from_file(self, file_path: str, references: list[list[str]] | None = None) -> dict[str, float] | None:
        with open(file_path, "r") as f:
            data = json.load(f)

        metrics = data.get("metrics", None)
        if not self.force and metrics is not None:
            return

        texts = data.get("text_samples", None)

        # Fallback: extract generations from math-style results list
        # Supports shapes: {"results": [...]}, {"results": {"results": [...]}}, [...]
        if texts is None:
            raw = data if isinstance(data, list) else data.get("results")
            if isinstance(raw, dict):
                raw = raw.get("results")
            if isinstance(raw, list):
                texts = [r["generations"] for r in raw if isinstance(r.get("generations"), list)]

        if not texts:
            print(f"Skipping {file_path}")
            return None

        metrics = self.evaluate(texts, references=references)

        data["metrics"] = metrics

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return metrics


class MathEvaluator:
    """Checks model generations against a known numeric answer.

    Uses a robust math/LaTeX post-processor to extract and canonicalize
    numeric answers from raw generations, then compares them against the
    expected answer.

    Example
    -------
    >>> ev = MathEvaluator()
    >>> ev.check("Step by step: 3 + 4 = 7", "7")
    1
    >>> ev.check("The answer is 42.", "7")
    0
    """

    def __init__(self, use_math_parser: bool = True):
        """Parameters
        ----------
        use_math_parser:
            If *True*, use the class-based universal parser. When *False*,
            fall back to the module-level universal helper.
        """
        self._use_math_parser = use_math_parser
        if use_math_parser:
            self._parser = MathParser()
        self._string_metrics = StringMetrics()

    def _extract(self, text: str) -> str:
        """Extract a normalised numeric string from *text*."""
        if self._use_math_parser:
            return self._parser.extract_universal_numeric(text)
        return universal_math_postprocess(text)

    def check(self, generation: str, answer_number: str) -> int:
        """Return 1 if *generation* contains *answer_number*, else 0.

        Parameters
        ----------
        generation:
            Raw text produced by the model.
        answer_number:
            The expected numeric answer (string, commas already stripped).
        """
        extracted = self._extract(generation)
        expected = self._extract(answer_number)
        if expected == "NULL":
            expected = answer_number.replace(",", "").strip()
        return int(extracted == expected)

    def score_group(self, generations: list[str], answer_number: str) -> list[int]:
        """Check a list of generations and return a list of 0/1 scores."""
        return [self.check(g, answer_number) for g in generations]

    def accuracy(self, generations: list[str], answer_number: str) -> float:
        """Fraction of *generations* that contain the correct answer."""
        scores = self.score_group(generations, answer_number)
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
        """Unbiased pass@k estimator (Chen et al., 2021 — HumanEval).

        Parameters
        ----------
        n : total number of samples for this problem.
        c : number of correct samples.
        k : the k in pass@k.

        Returns the probability that at least one of *k* randomly drawn
        (without replacement) samples is correct.
        """
        if n < k:
            return float("nan")
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        # 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
        num = 1.0
        den = 1.0
        for i in range(k):
            num *= n - c - i
            den *= n - i
        return 1.0 - num / den

    def evaluate(  # noqa: C901, PLR0912
        self,
        generations: list[list[str]],
        gold_answers: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float | str]:
        """Compute comprehensive math evaluation metrics.

        Parameters
        ----------
        generations:
            Outer list is per-question; inner list contains the model's
            sampled generations for that question.
        gold_answers:
            One gold numeric answer string per question.
        k_values:
            Which k's to compute pass@k for. Defaults to [1, 2, 4, 8, 16]
            clipped to the actual group size. Duplicates and out-of-range
            values are silently removed.

        Returns
        -------
        Flat dict of metric_name → float, plus a ``math_metrics_summary``
        string formatted like the ``Evaluator.evaluate()`` summaries.
        """
        if not generations:
            return {}

        group_size = max(len(g) for g in generations)

        if k_values is None:
            k_values = [1, 2, 4, 8, 16]
        # Clamp to valid range and deduplicate, preserving order
        seen: set[int] = set()
        effective_ks: list[int] = []
        for k in k_values:
            if 1 <= k <= group_size and k not in seen:
                effective_ks.append(k)
                seen.add(k)

        # ── per-question correctness ──────────────────────────────────────
        per_question_acc: list[float] = []
        pass_at_k_per_q: dict[int, list[float]] = {k: [] for k in effective_ks}

        for gens, gold in zip(generations, gold_answers):
            if not gens:
                continue
            scores = self.score_group(gens, gold)
            n = len(scores)
            c = sum(scores)
            per_question_acc.append(c / n)
            for k in effective_ks:
                pass_at_k_per_q[k].append(self._pass_at_k_estimator(n, c, k))

        metrics: dict[str, float | str] = {}

        # Accuracy with full statistics (mean, CI, etc.)
        metrics.update(compute_statistics(per_question_acc, "accuracy"))

        # pass@k — report mean across questions (filter NaN from k > n cases)
        for k in effective_ks:
            vals = [v for v in pass_at_k_per_q[k] if not np.isnan(v)]
            metrics[f"pass_at_{k}"] = float(np.mean(vals)) if vals else float("nan")

        metrics["k"] = float(group_size)

        # ── string metrics ────────────────────────────────────────────────
        # Gold answers are the single reference for each question.
        references: list[list[str]] = [[g] for g in gold_answers]
        string_stats = self._string_metrics(generations, references)
        metrics.update(string_stats)

        # ── batch-level diversity ─────────────────────────────────────────
        # Flatten all generations across questions and compute diversity over
        # the entire batch (complements the per-group averages above).
        all_generations_flat: list[str] = [g for group in generations for g in group]
        all_references_flat: list[str] = [r for refs in references for r in refs]
        batch_diversity = self._string_metrics.compute_diversity_batch(
            all_generations_flat,
            references_for_vocab=all_references_flat if all_references_flat else None,
            prefix="batch",
        )
        metrics.update(batch_diversity)

        # ── summary string ────────────────────────────────────────────────
        summary_parts: list[str] = []

        # Accuracy with CI
        acc_mean = metrics.get("accuracy", float("nan"))
        acc_ci = metrics.get("accuracy_ci95", float("nan"))
        summary_parts.append(f"Acc: {_format_summary_value(acc_mean, acc_ci)}")

        # pass@k values
        for k in effective_ks:
            val = metrics.get(f"pass_at_{k}", float("nan"))
            if not np.isnan(val):
                summary_parts.append(f"pass@{k}: {_format_num(val)}")

        # Per-group string metrics with CI
        for key, display_name in [
            ("f1", "F1"),
            ("bleu", "BLEU"),
            ("distinct_2", "Dist-2"),
            ("self_bleu", "S-BLEU"),
        ]:
            if key in metrics:
                ci_key = f"{key}_ci95"
                if ci_key in metrics:
                    summary_parts.append(f"{display_name}: {_format_summary_value(metrics[key], metrics[ci_key])}")
                else:
                    summary_parts.append(f"{display_name}: {_format_num(metrics[key])}")

        # Batch-level diversity metrics (single values, no CI)
        for key, display_name in [
            ("batch_distinct_2", "B-Dist2"),
            ("batch_self_bleu", "B-S-BLEU"),
        ]:
            val = metrics.get(key, float("nan"))
            if not (isinstance(val, float) and np.isnan(val)):
                summary_parts.append(f"{display_name}: {_format_num(val)}")

        if summary_parts:
            metrics["math_metrics_summary"] = " | ".join(summary_parts)

        return metrics

    def eval_from_file(
        self,
        file_path: str,
        force: bool = False,
        k_values: list[int] | None = None,
    ) -> dict[str, float | str] | None:
        """Load a math results JSON file (as produced by ``llada_math.py``)
        and compute (or re-compute) evaluation metrics in-place.

        Supported JSON shapes
        ---------------------
        1. ``[{question, gold_answer, generations, ...}, ...]`` — plain list at root
        2. ``{"results": [{...}, ...], ...}`` — normal final output
        3. ``{"results": {"results": [...], ...}, ...}`` — temp checkpoint

        Only ``generations`` (list[str]) is strictly required per entry;
        ``gold_answer`` (str) defaults to ``""`` if absent.
        Computed metrics are written back as ``"math_metrics"``.

        Parameters
        ----------
        file_path: path to the JSON file.
        force: if False and ``"math_metrics"`` already present, skip.
        k_values: forwarded to :meth:`evaluate`.
        """
        with open(file_path) as f:
            data = json.load(f)

        # ── normalise root to a list of result dicts ──────────────────────
        if isinstance(data, list):
            # Shape 1: root is already a list of result dicts
            results: list[dict] = data
            data = {"results": results}  # re-wrap so we can write metrics back
        else:
            if not force and data.get("math_metrics") is not None:
                return data["math_metrics"]

            results = data.get("results")
            if results is None:
                return None

            # Shape 3: {"results": {"results": [...], ...}, ...}
            if isinstance(results, dict):
                results = results.get("results", [])

        if not isinstance(results, list):
            return None

        # ── extract fields; only generations is strictly required ─────────
        generations: list[list[str]] = []
        gold_answers: list[str] = []
        for r in results:
            gens = r.get("generations")
            if not isinstance(gens, list):
                continue  # skip malformed entries
            generations.append(gens)
            gold_answers.append(str(r.get("gold_answer", "")))

        if not generations:
            return None

        math_metrics = self.evaluate(generations, gold_answers, k_values=k_values)
        data["math_metrics"] = math_metrics

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return math_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate text samples.")
    parser.add_argument(
        "--folder_path",
        "-f",
        type=str,
        required=True,
        help="Path to the folder containing text samples.",
    )
    parser.add_argument("--ppl_model_id", type=str, default="gpt2", help="Model ID for perplexity calculation.")
    parser.add_argument(
        "--cos_model_id",
        type=str,
        default="jinaai/jina-embeddings-v2-base-en",
        help="Model ID for cosine similarity calculation.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="Batch size for evaluation.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if metrics exist.")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.folder_path) if f.endswith(".json") and not f.startswith("temp")]
    evaluator = Evaluator(args.batch_size, args.force, args.ppl_model_id, args.cos_model_id)
    pbar = tqdm(files, desc="Evaluating files")

    for file_name in pbar:
        file_path = os.path.join(args.folder_path, file_name)
        evaluator.eval_from_file(file_path)


if __name__ == "__main__":
    main()

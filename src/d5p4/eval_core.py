"""
Core evaluation module for text generation metrics.

Metric classes: Perplexity, AverageCosineSimilarity, MAUVE,
WassersteinDistance, StringMetrics.

High-level evaluators: Evaluator (generation quality), MathEvaluator
(math correctness + string overlap).

Utility helpers live in eval_utils.py.
"""

import argparse
import json
import os

import numpy as np
import ot
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2Model, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from d5p4 import mauve
from d5p4.config import CACHE_DIR
from d5p4.eval_utils import (
    _BATCH_SELF_BLEU_EXACT_THRESHOLD,
    _compute_distinct_metrics_impl,
    _compute_f1_score,
    _compute_self_bleu_bounded_impl,
    _compute_self_bleu_impl,
    _emit_timing_summary,
    _format_asymmetric_ci,
    _format_num,
    _format_summary_value,
    _get_corpus_bleu_metric,
    _group_diversity_task,
    _group_reference_alignment_task,
    _is_math_results_file,
    _map_tasks,
    _math_group_task,
    _time_call,
    _vocab_size_from_refs,
    compute_statistics,
)
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.text_postprocessors import MathParser, universal_math_postprocess
from d5p4.utils import print as u_print
from d5p4.utils import process_model_args, tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metric modules
# ---------------------------------------------------------------------------


class Perplexity(torch.nn.Module):
    def __init__(self, model: AutoModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenizer_pad_id = self.tokenizer.pad_token_id
        assert isinstance(tokenizer_pad_id, int)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer_pad_id)

        if isinstance(self.model, GPT2Model):
            self.lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
            self.lm_head.weight = self.model.wte.weight  # tie weights
        elif isinstance(self.model, LlamaForCausalLM):
            self.lm_head = self.model.lm_head
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _forward(self, texts: list[str]) -> list[float] | None:
        """Compute per-sample mean NLL values. Statistics should be computed in this space."""
        texts = [t.strip() for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

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

            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

        loss = loss.clamp(max=15.0) * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_loss = loss.sum(dim=1) / token_counts
        mean_loss = torch.nan_to_num(mean_loss, nan=15.0, posinf=15.0, neginf=0.0)
        return mean_loss.cpu().tolist()

    def forward(self, texts: list[list[str]], batch_size: int = 0) -> dict[str, float]:
        """Compute perplexity statistics.

        Statistics are computed in NLL space, then transformed via ``exp()``.
        """
        flattened_texts = [text for sublist in texts for text in sublist]
        batch_size = batch_size or len(flattened_texts)

        nlls: list[float] = []
        for start in range(0, len(flattened_texts), batch_size):
            result = self._forward(flattened_texts[start : start + batch_size])
            if result is not None:
                nlls.extend(result)

        nll_stats = compute_statistics(nlls, "nll")

        return {
            "perplexity": np.exp(nll_stats["nll"]),
            "perplexity_mean": np.exp(nll_stats["nll_mean"]),
            "perplexity_median": np.exp(nll_stats["nll_median"]),
            "perplexity_min": np.exp(nll_stats["nll_min"]),
            "perplexity_max": np.exp(nll_stats["nll_max"]),
            "perplexity_ci95_lower": np.exp(nll_stats["nll_mean"] - nll_stats["nll_ci95"]),
            "perplexity_ci95_upper": np.exp(nll_stats["nll_mean"] + nll_stats["nll_ci95"]),
            "perplexity_count": nll_stats["nll_count"],
        }


class AverageCosineSimilarity(torch.nn.Module):
    def __init__(self, model: JinaBertModel):
        super().__init__()
        self.model = model

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to L2-normalised embeddings."""
        self.model.to(device)
        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)  # type: ignore
            x = embeddings.reshape(len(texts), -1)
            x = F.normalize(x, p=2, dim=-1)
        return x

    def _forward(self, texts: list[str]) -> float:
        if isinstance(texts, str):
            texts = [texts]

        x = self._encode(texts)
        S = torch.mm(x, x.t()) - torch.eye(len(texts), device=x.device)

        n = S.size(0)
        if n <= 1:
            return 0.0
        return (S.sum() / (n * (n - 1))).item()

    def compute_max_alignment(
        self,
        predictions: list[list[str]],
        references: list[list[str]],
    ) -> list[float]:
        """For each group, compute the max cosine similarity between any prediction and any reference."""
        max_alignments = []
        for preds, refs in zip(predictions, references):
            if not preds or not refs:
                max_alignments.append(0.0)
                continue
            embeddings = self._encode(preds + refs)
            pred_embs = embeddings[: len(preds)]
            ref_embs = embeddings[len(preds) :]
            max_alignments.append(torch.mm(pred_embs, ref_embs.t()).max().item())
        return max_alignments

    def forward(self, texts: list[list[str]]) -> dict[str, float]:
        """Compute average cosine similarity statistics across groups."""
        avg_cos_sims = [self._forward(group) for group in texts]
        return compute_statistics(avg_cos_sims, "cosine_similarity")


class MAUVE(torch.nn.Module):
    def __init__(self, model: AutoModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, p_text: list[str], q_text: list[str]):
        """Compute MAUVE score using the mauve package."""
        return mauve.compute_mauve(
            p_text=p_text,
            q_text=q_text,
            models=(self.model, self.tokenizer),
            device_id=0 if torch.cuda.is_available() else -1,
        )


class WassersteinDistance(torch.nn.Module):
    def __init__(self, model: JinaBertModel):
        super().__init__()
        self.model = model

    def _encode(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            embeddings: torch.Tensor = self.model.encode(texts, convert_to_tensor=True, device=device)  # type: ignore
            x = embeddings.reshape(len(texts), -1)
            x = F.normalize(x, p=2, dim=-1)
        return x.cpu()

    def forward(
        self,
        generations: list[str],
        good_references: list[str],
        bad_references: list[str] | None = None,
    ) -> tuple[float, float]:
        n_gen = len(generations)
        n_good = len(good_references)

        all_texts = generations + good_references + (bad_references or [])
        embeddings = self._encode(all_texts).numpy()

        gen_embs = embeddings[:n_gen]
        good_embs = embeddings[n_gen : n_gen + n_good]

        p_gen = np.ones(n_gen) / n_gen
        p_good = np.ones(n_good) / n_good
        wasserstein_good: float = ot.emd2(p_gen, p_good, ot.dist(gen_embs, good_embs, metric="euclidean"))  # type: ignore

        wasserstein_bad = float("nan")
        if bad_references:
            n_bad = len(bad_references)
            bad_embs = embeddings[n_gen + n_good :]
            p_bad = np.ones(n_bad) / n_bad
            wasserstein_bad: float = ot.emd2(p_gen, p_bad, ot.dist(gen_embs, bad_embs, metric="euclidean"))  # type: ignore

        return wasserstein_good, wasserstein_bad


class StringMetrics(torch.nn.Module):
    """Lexical string metrics: distinct-n, empirical entropy, self-BLEU, F1, BLEU vs references."""

    def __init__(self):
        super().__init__()

    def diversity_set(
        self,
        texts: list[str],
        references_for_vocab: list[str] | None = None,
        prefix: str = "batch",
        vocab_size: int | None = None,
        bounded_self_bleu: bool = False,
    ) -> dict[str, float]:
        """Compute lexical diversity metrics over a single flat set of texts.

        Parameters
        ----------
        texts:
            Texts to evaluate.
        references_for_vocab:
            Optional strings used to estimate vocabulary size for EAD.
        prefix:
            Key prefix for the returned metric dictionary.
        bounded_self_bleu:
            When ``True``, uses a reference-capped approximation of self-BLEU
            suitable for large corpora.
        """
        if not texts:
            return {}

        if vocab_size is None:
            vocab_size = _vocab_size_from_refs(references_for_vocab)
        distinct_metrics = _compute_distinct_metrics_impl(texts, vocab_size=vocab_size)
        self_bleu = _compute_self_bleu_bounded_impl(texts) if bounded_self_bleu else _compute_self_bleu_impl(texts)

        return {f"{prefix}_{k}": v for k, v in distinct_metrics.items()} | {f"{prefix}_self_bleu": self_bleu}

    def diversity_grouped(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
        num_workers: int = 1,
    ) -> dict[str, float]:
        """Compute per-group lexical diversity, then aggregate statistics across groups."""
        ref_tokens = [s for refs in references for s in refs] if references and any(references) else None
        vocab_size = _vocab_size_from_refs(ref_tokens)
        valid_groups = [group for group in predictions if group]

        distinct_metrics_list: list[dict[str, float]] = []
        self_bleu_scores: list[float] = []
        tasks = [(group, vocab_size) for group in valid_groups]
        for d_metrics, self_bleu in _map_tasks(_group_diversity_task, tasks, num_workers):
            distinct_metrics_list.append(d_metrics)
            self_bleu_scores.append(self_bleu)

        all_metrics: dict[str, float] = {}
        if distinct_metrics_list:
            for key in distinct_metrics_list[0]:
                values = [m[key] for m in distinct_metrics_list if key in m]
                all_metrics.update(compute_statistics(values, key))
        all_metrics.update(compute_statistics(self_bleu_scores, "self_bleu"))
        return all_metrics

    def diversity_corpus(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
        prefix: str = "batch",
    ) -> dict[str, float]:
        """Compute lexical diversity over the full flattened generations corpus."""
        all_generations_flat = [g for group in predictions for g in group]
        if not all_generations_flat:
            return {}

        vocab_size = None
        if references and any(refs for refs in references):
            vocab_size = _vocab_size_from_refs([r for refs in references for r in refs])

        return self.diversity_set(
            all_generations_flat,
            prefix=prefix,
            vocab_size=vocab_size,
            bounded_self_bleu=len(all_generations_flat) > _BATCH_SELF_BLEU_EXACT_THRESHOLD,
        )

    def reference_alignment(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
        num_workers: int = 1,
    ) -> dict[str, float]:
        """Compute lexical overlap metrics against references (F1, BLEU)."""
        if not (references and any(refs for refs in references)):
            return {}

        grouped_pairs = [(preds, refs) for preds, refs in zip(predictions, references) if preds and refs]
        flattened_predictions = [pred for preds, _ in grouped_pairs for pred in preds]
        flattened_references = [refs for preds, refs in grouped_pairs for _ in preds]

        f1_scores: list[float] = []
        f1_at_k_scores: list[float] = []
        bleu_at_k_scores: list[float] = []
        for group_f1_scores, best_f1, best_bleu in _map_tasks(
            _group_reference_alignment_task,
            grouped_pairs,
            num_workers,
        ):
            f1_scores.extend(group_f1_scores)
            f1_at_k_scores.append(best_f1)
            bleu_at_k_scores.append(best_bleu)

        all_metrics: dict[str, float] = {}
        all_metrics.update(compute_statistics(f1_scores, "f1"))

        bleu_score = 0.0
        if flattened_references:
            max_refs = max(len(refs) for refs in flattened_references)
            formatted_refs = [
                [refs[i] if i < len(refs) else refs[0] for refs in flattened_references] for i in range(max_refs)
            ]
            bleu_score = _get_corpus_bleu_metric().corpus_score(flattened_predictions, formatted_refs).score

        all_metrics["bleu"] = bleu_score
        k = len(predictions[0]) if predictions and predictions[0] else 0
        all_metrics["k"] = float(k)
        if k > 0:
            all_metrics.update(compute_statistics(f1_at_k_scores, "f1_at_k"))
            all_metrics.update(compute_statistics(bleu_at_k_scores, "bleu_at_k"))

        return all_metrics

    def forward(
        self,
        predictions: list[list[str]],
        references: list[list[str]] | None = None,
        num_workers: int = 1,
    ) -> dict[str, float]:
        """Backward-compatible wrapper: reference alignment + grouped diversity."""
        return {
            **self.reference_alignment(predictions, references, num_workers=num_workers),
            **self.diversity_grouped(predictions, references, num_workers=num_workers),
        }


# ---------------------------------------------------------------------------
# High-level evaluators
# ---------------------------------------------------------------------------


def _build_perplexity_model(model_id: str) -> Perplexity:
    """Load a causal-LM backbone and wrap it in a :class:`Perplexity` scorer."""
    args = process_model_args(model_id, cache_dir=CACHE_DIR)
    model = LlamaForCausalLM.from_pretrained(**args) if "llama" in model_id else AutoModel.from_pretrained(**args)
    tokenizer = AutoTokenizer.from_pretrained(**args)
    return Perplexity(model, tokenizer)


class Evaluator:
    """Generation-quality evaluator wrapping Perplexity, CosineSimilarity,
    MAUVE, WassersteinDistance, and StringMetrics."""

    def __init__(
        self,
        batch_size: int = 0,
        force: bool = False,
        ppl_model_id: str = "gpt2",
        cos_model_id: str = "jinaai/jina-embeddings-v2-base-en",
        show_timings: bool = False,
    ):
        self.perplexity_model = _build_perplexity_model(ppl_model_id)
        self.mauve_model = MAUVE(self.perplexity_model.model, self.perplexity_model.tokenizer)  # reuse backbone

        cos_models_args = process_model_args(cos_model_id, cache_dir=CACHE_DIR)
        cos_model = JinaBertModel.from_pretrained(**cos_models_args)
        self.cosine_model = AverageCosineSimilarity(cos_model)
        self.wasserstein_model = WassersteinDistance(cos_model)  # reuse embedding model for WD
        self.string_metrics = StringMetrics()

        self.batch_size = batch_size
        self.force = force
        self.show_timings = show_timings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, texts: list[list[str]], references: list[list[str]] | None = None) -> dict[str, float | str]:
        timings: list[tuple[str, float]] = []

        ppl_stats, elapsed = _time_call(self.perplexity_model, texts, batch_size=self.batch_size)
        timings.append(("perplexity", elapsed))

        cos_stats, elapsed = _time_call(self.cosine_model, texts)
        timings.append(("cosine_similarity", elapsed))

        string_stats, elapsed = _time_call(self.compute_string_metrics, texts, references=references)
        timings.append(("string_metrics", elapsed))

        wd_stats: dict[str, float] = {}
        if references and any(refs for refs in references):

            def _compute_wd_stats() -> dict[str, float]:
                wd_scores = []
                assert references is not None
                for group_gen, group_ref in zip(texts, references):
                    if not group_gen or not group_ref:
                        continue
                    wd_good, _ = self.wasserstein_model(group_gen, group_ref, bad_references=None)
                    wd_scores.append(wd_good)
                return compute_statistics(wd_scores, "wasserstein_distance")

            wd_stats, elapsed = _time_call(_compute_wd_stats)
            timings.append(("wasserstein_distance", elapsed))

        metrics = {**ppl_stats, **cos_stats, **string_stats, **wd_stats}

        # Build human-readable summary
        summary_targets = [
            ("perplexity", "PPL"),
            ("cosine_similarity", "CosSim"),
            ("wasserstein_distance", "WD"),
            ("distinct_2", "Dist-2"),
            ("empirical_entropy", "Ent"),
            ("self_bleu", "S-BLEU"),
            ("cos_at_k", "Cos@k"),
        ]
        summary_parts = []
        for key, display_name in summary_targets:
            if key == "perplexity" and f"{key}_ci95_lower" in metrics:
                val_str = _format_asymmetric_ci(
                    metrics[key],
                    metrics[f"{key}_ci95_lower"],
                    metrics[f"{key}_ci95_upper"],
                )
                summary_parts.append(f"{display_name}: {val_str}")
            elif key in metrics and f"{key}_ci95" in metrics:
                summary_parts.append(f"{display_name}: {_format_summary_value(metrics[key], metrics[f'{key}_ci95'])}")

        if summary_parts:
            metrics["metrics_summary"] = " | ".join(summary_parts)

        if self.show_timings:
            _emit_timing_summary("evaluator", timings)

        return metrics

    def compute_mauve(self, references: list[str], generations: list[str]) -> float:
        return self.mauve_model(references, generations).mauve

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
        timings: list[tuple[str, float]] = []

        reference_metrics, elapsed = _time_call(self.string_metrics.reference_alignment, predictions, references)
        timings.append(("reference_alignment", elapsed))

        diversity_metrics, elapsed = _time_call(self.string_metrics.diversity_grouped, predictions, references)
        timings.append(("diversity_grouped", elapsed))

        metrics = {**reference_metrics, **diversity_metrics}

        if references and any(refs for refs in references):
            cos_at_k_scores, elapsed = _time_call(self.cosine_model.compute_max_alignment, predictions, references)
            timings.append(("cos_at_k", elapsed))
            metrics.update(compute_statistics(cos_at_k_scores, "cos_at_k"))

        if self.show_timings:
            _emit_timing_summary("string_metrics", timings)

        return metrics

    def evaluate_baseline(  # noqa: C901
        self,
        full_sequences: list[list[str]],
        metric: str,
        k: int,
        references: list[list[str]] | None = None,
    ) -> list[list[str]]:
        """Select the *k* best sequences per group according to *metric*.

        Supported metrics
        -----------------
        ``"ppl"``
            Lower is better.
        ``"f1"``
            Higher is better. Requires *references*.
        """
        flattened_texts = [text for sublist in full_sequences for text in sublist]
        group_sizes = [len(sublist) for sublist in full_sequences]

        def unflatten(flat_list: list) -> list[list]:
            out, cursor = [], 0
            for size in group_sizes:
                out.append(flat_list[cursor : cursor + size])
                cursor += size
            return out

        if metric.lower() == "ppl":
            batch_size = self.batch_size or len(flattened_texts)
            nlls: list[float] = []
            for start in range(0, len(flattened_texts), batch_size):
                result = self.perplexity_model._forward(flattened_texts[start : start + batch_size])
                if result is not None:
                    nlls.extend(result)
                else:
                    u_print("Skipping batch of empty texts", flattened_texts[start : start + batch_size])
            unflattened_scores = unflatten(nlls)
            reverse_sort = False

        elif metric.lower() == "f1":
            if references is None:
                raise ValueError("References must be provided for the f1 metric.")
            unflattened_scores = [
                [max((_compute_f1_score(cand, ref) for ref in group_refs), default=0.0) for cand in group_cands]
                for group_cands, group_refs in zip(full_sequences, references)
            ]
            reverse_sort = True

        else:
            raise ValueError(f"Metric '{metric}' not supported. Choose 'ppl' or 'f1'.")

        selected_sequences = []
        for group_texts, group_scores in zip(full_sequences, unflattened_scores):
            top_k_indices = sorted(
                [idx for idx, _ in sorted(enumerate(group_scores), key=lambda x: x[1], reverse=reverse_sort)[:k]],
            )
            selected_sequences.append([group_texts[idx] for idx in top_k_indices])

        return selected_sequences

    def eval_from_file(
        self,
        file_path: str,
        references: list[list[str]] | None = None,
    ) -> dict[str, float | str] | None:
        with open(file_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", None)
        if not self.force and metrics is not None:
            return

        texts = data.get("text_samples", None)

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
    """Check model generations against a known numeric answer.

    Uses a robust math/LaTeX post-processor to extract and canonicalize
    numeric answers, then compares them against the expected answer.

    Example
    -------
    >>> ev = MathEvaluator()
    >>> ev.check("Step by step: 3 + 4 = 7", "7")
    1
    >>> ev.check("The answer is 42.", "7")
    0
    """

    def __init__(
        self,
        use_math_parser: bool = True,
        show_timings: bool = False,
        ppl_model_id: str | None = None,
        cos_model_id: str | None = None,
        batch_size: int = 0,
    ):
        """
        Parameters
        ----------
        use_math_parser:
            If ``True``, use the class-based universal parser. When ``False``,
            fall back to the module-level universal helper.
        ppl_model_id:
            When provided, perplexity is computed over all generations.
        cos_model_id:
            When provided, per-group cosine similarity and cos@k vs references
            are computed.
        batch_size:
            Batch size for perplexity forward passes.
        """
        self._use_math_parser = use_math_parser
        if use_math_parser:
            self._parser = MathParser()
        self._string_metrics = StringMetrics()
        self.show_timings = show_timings
        self.batch_size = batch_size

        self._perplexity_model: Perplexity | None = None
        self._cosine_model: AverageCosineSimilarity | None = None

        if ppl_model_id is not None:
            self._perplexity_model = _build_perplexity_model(ppl_model_id)

        if cos_model_id is not None:
            cos_models_args = process_model_args(cos_model_id, cache_dir=CACHE_DIR)
            cos_model = JinaBertModel.from_pretrained(**cos_models_args)
            self._cosine_model = AverageCosineSimilarity(cos_model)

    # ------------------------------------------------------------------
    # Low-level correctness checks
    # ------------------------------------------------------------------

    def _extract(self, text: str) -> str:
        """Extract a normalised numeric string from *text*."""
        if self._use_math_parser:
            return self._parser.extract_universal_numeric(text)
        return universal_math_postprocess(text)

    def check(self, generation: str, answer_number: str) -> int:
        """Return 1 if *generation* contains the correct answer, else 0."""
        extracted = self._extract(generation)
        expected = self._extract(answer_number)
        if expected == "NULL":
            expected = answer_number.replace(",", "").strip()
        return int(extracted == expected)

    def score_group(self, generations: list[str], answer_number: str) -> list[int]:
        """Return a 0/1 score for each generation."""
        return [self.check(g, answer_number) for g in generations]

    def accuracy(self, generations: list[str], answer_number: str) -> float:
        """Fraction of *generations* that contain the correct answer."""
        scores = self.score_group(generations, answer_number)
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
        """Unbiased pass@k estimator (Chen et al., 2021).

        Parameters
        ----------
        n: total samples for this problem.
        c: number of correct samples.
        k: the k in pass@k.
        """
        if n < k:
            return float("nan")
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        num, den = 1.0, 1.0
        for i in range(k):
            num *= n - c - i
            den *= n - i
        return 1.0 - num / den

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        generations: list[list[str]],
        gold_answers: list[str],
        string_references: list[list[str]] | None = None,
        k_values: list[int] | None = None,
        num_workers: int = 1,
        batch_size: int | None = None,
    ) -> dict[str, float | str]:
        """Compute comprehensive math evaluation metrics.

        Parameters
        ----------
        generations:
            Per-question list of sampled model outputs.
        gold_answers:
            One gold numeric answer string per question.
        string_references:
            Optional per-question reference strings for F1/BLEU.
            Falls back to ``gold_answers`` when omitted.
        k_values:
            Which k values to compute pass@k for. Defaults to [1, 2, 4, 8, 16],
            clamped to the actual group size.
        batch_size:
            Override the instance-level batch size for this call.

        Returns
        -------
        Flat ``dict`` of metric → value, plus a ``math_metrics_summary`` string.
        """
        if not generations:
            return {}

        group_size = max(len(g) for g in generations)

        if k_values is None:
            k_values = [1, 2, 4, 8, 16]
        seen: set[int] = set()
        effective_ks: list[int] = []
        for k in k_values:
            if 1 <= k <= group_size and k not in seen:
                effective_ks.append(k)
                seen.add(k)

        timings: list[tuple[str, float]] = []

        # ── correctness / pass@k ──────────────────────────────────────────
        per_question_acc: list[float] = []
        pass_at_k_per_q: dict[int, list[float]] = {k: [] for k in effective_ks}
        valid_groups = [(gens, gold) for gens, gold in zip(generations, gold_answers) if gens]

        def _run_correctness() -> None:
            tasks = [(gens, gold, effective_ks, self._use_math_parser) for gens, gold in valid_groups]
            for acc, pass_at_k in _map_tasks(_math_group_task, tasks, num_workers):
                per_question_acc.append(acc)
                for k in effective_ks:
                    pass_at_k_per_q[k].append(pass_at_k[k])

        _, elapsed = _time_call(_run_correctness)
        timings.append(("correctness_pass@k", elapsed))

        metrics: dict[str, float | str] = {}
        metrics.update(compute_statistics(per_question_acc, "accuracy"))

        for k in effective_ks:
            vals = [v for v in pass_at_k_per_q[k] if not np.isnan(v)]
            pass_k_stats = compute_statistics(vals, f"pass_at_{k}")
            metrics.update(pass_k_stats)
            metrics[f"pass@{k}"] = pass_k_stats[f"pass_at_{k}"]

        metrics["k"] = float(group_size)

        # ── string metrics ────────────────────────────────────────────────
        references = string_references if string_references is not None else [[g] for g in gold_answers]
        string_stats, elapsed = _time_call(self._string_metrics, generations, references, num_workers=num_workers)
        timings.append(("string_metrics", elapsed))
        metrics.update(string_stats)

        # ── batch-level diversity ─────────────────────────────────────────
        batch_diversity, elapsed = _time_call(
            self._string_metrics.diversity_corpus,
            generations,
            references=references,
            prefix="batch",
        )
        timings.append(("batch_diversity", elapsed))
        metrics.update(batch_diversity)

        # ── neural metrics (perplexity + cosine) ──────────────────────────
        _batch_size = batch_size if batch_size is not None else self.batch_size
        if self._perplexity_model is not None:
            ppl_stats, elapsed = _time_call(
                self._perplexity_model,
                generations,
                batch_size=_batch_size,
            )
            timings.append(("perplexity", elapsed))
            metrics.update(ppl_stats)

        if self._cosine_model is not None:
            cos_stats, elapsed = _time_call(self._cosine_model, generations)
            timings.append(("cosine_similarity", elapsed))
            metrics.update(cos_stats)

            if references and any(refs for refs in references):
                cos_at_k_scores, elapsed = _time_call(
                    self._cosine_model.compute_max_alignment,
                    generations,
                    references,
                )
                timings.append(("cos_at_k", elapsed))
                metrics.update(compute_statistics(cos_at_k_scores, "cos_at_k"))

        # ── summary string ────────────────────────────────────────────────
        summary_parts: list[str] = []

        acc_mean = metrics.get("accuracy", float("nan"))
        acc_ci = metrics.get("accuracy_ci95", float("nan"))
        summary_parts.append(f"Acc: {_format_summary_value(acc_mean, acc_ci)}")

        for k in effective_ks:
            val = metrics.get(f"pass_at_{k}", float("nan"))
            ci = metrics.get(f"pass_at_{k}_ci95", float("nan"))
            if not np.isnan(val):
                summary_parts.append(f"pass@{k}: {_format_summary_value(val, ci)}")

        # Perplexity (asymmetric CI in PPL space)
        if "perplexity" in metrics and "perplexity_ci95_lower" in metrics:
            ppl_mean = metrics["perplexity"]
            ppl_lower = metrics["perplexity_ci95_lower"]
            ppl_upper = metrics["perplexity_ci95_upper"]
            assert isinstance(ppl_mean, float) and isinstance(ppl_lower, float) and isinstance(ppl_upper, float)
            val_str = _format_asymmetric_ci(
                ppl_mean,
                ppl_lower,
                ppl_upper,
            )
            summary_parts.append(f"PPL: {val_str}")

        # Cosine similarity (per-group)
        if "cosine_similarity" in metrics and "cosine_similarity_ci95" in metrics:
            summary_parts.append(
                f"CosSim: {_format_summary_value(metrics['cosine_similarity'], metrics['cosine_similarity_ci95'])}",
            )

        # Cos@k (vs references)
        if "cos_at_k" in metrics and "cos_at_k_ci95" in metrics:
            summary_parts.append(
                f"Cos@k: {_format_summary_value(metrics['cos_at_k'], metrics['cos_at_k_ci95'])}",
            )

        for key, display_name in [
            ("f1", "F1"),
            ("bleu", "BLEU"),
            ("distinct_2", "Dist-2"),
            ("empirical_entropy", "Ent"),
            ("self_bleu", "S-BLEU"),
        ]:
            if key in metrics:
                ci_key = f"{key}_ci95"
                if ci_key in metrics:
                    summary_parts.append(f"{display_name}: {_format_summary_value(metrics[key], metrics[ci_key])}")
                else:
                    summary_parts.append(f"{display_name}: {_format_num(metrics[key])}")

        for key, display_name in [
            ("batch_distinct_2", "B-Dist2"),
            ("batch_empirical_entropy", "B-Ent"),
            ("batch_self_bleu", "B-S-BLEU"),
        ]:
            val = metrics.get(key, float("nan"))
            if not (isinstance(val, float) and np.isnan(val)):
                summary_parts.append(f"{display_name}: {_format_num(val)}")

        if summary_parts:
            metrics["math_metrics_summary"] = " | ".join(summary_parts)

        if self.show_timings:
            _emit_timing_summary("math_evaluator", timings)

        return metrics

    def eval_from_file(
        self,
        file_path: str,
        force: bool = False,
        k_values: list[int] | None = None,
        num_workers: int = 1,
    ) -> dict[str, float | str] | None:
        """Load a math results JSON file and compute (or refresh) metrics in-place.

        Supported JSON shapes
        ---------------------
        1. ``[{question, gold_answer, generations, ...}, ...]`` — plain list at root
        2. ``{"results": [{...}, ...], ...}`` — normal final output
        3. ``{"results": {"results": [...], ...}, ...}`` — temp checkpoint

        Computed metrics are written back as ``"math_metrics"``.
        """
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            results: list[dict] = data
            data = {"results": results}
        else:
            if not force and data.get("math_metrics") is not None:
                return data["math_metrics"]
            results = data.get("results")
            if results is None:
                return None
            if isinstance(results, dict):
                results = results.get("results", [])

        if not isinstance(results, list):
            return None

        generations: list[list[str]] = []
        gold_answers: list[str] = []
        string_references: list[list[str]] = []
        for r in results:
            gens = r.get("generations")
            if not isinstance(gens, list):
                continue
            generations.append(gens)
            gold_answers.append(str(r.get("gold_answer", "")))
            answer_str = r.get("answer_str")
            string_references.append(
                [answer_str] if isinstance(answer_str, str) and answer_str else [str(r.get("gold_answer", ""))],
            )

        if not generations:
            return None

        math_metrics = self.evaluate(
            generations,
            gold_answers,
            string_references=string_references,
            k_values=k_values,
            num_workers=num_workers,
        )
        data["math_metrics"] = math_metrics

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return math_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    n_cpus = os.cpu_count()
    assert n_cpus is not None

    parser = argparse.ArgumentParser(description="Evaluate text samples.")
    parser.add_argument(
        "--input_path",
        "--folder_path",
        "-i",
        "-f",
        type=str,
        required=True,
        help="Path to folder or single JSON file containing result samples.",
    )
    parser.add_argument("--ppl_model_id", type=str, default="gpt2", help="Model ID for perplexity calculation.")
    parser.add_argument(
        "--cos_model_id",
        type=str,
        default="jinaai/jina-embeddings-v2-base-en",
        help="Model ID for cosine similarity.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="Batch size for perplexity evaluation.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=n_cpus // 2,
        help="CPU workers for string/math metric aggregation.",
    )
    parser.add_argument("--force", action="store_true", help="Re-evaluate even when metrics already exist.")
    args = parser.parse_args()
    input_path = args.input_path
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json") and not f.startswith("temp")
        ]
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        return

    evaluator: Evaluator | None = None
    math_evaluator: MathEvaluator | None = None

    for file_path in tqdm(files, desc="Evaluating files"):
        if _is_math_results_file(file_path):
            if math_evaluator is None:
                math_evaluator = MathEvaluator(
                    show_timings=True,
                    ppl_model_id=args.ppl_model_id,
                    cos_model_id=args.cos_model_id,
                    batch_size=args.batch_size,
                )
            math_evaluator.eval_from_file(file_path, force=args.force, num_workers=args.num_workers)
        else:
            if evaluator is None:
                evaluator = Evaluator(
                    args.batch_size,
                    args.force,
                    args.ppl_model_id,
                    args.cos_model_id,
                    show_timings=True,
                )
            evaluator.eval_from_file(file_path)


if __name__ == "__main__":
    main()

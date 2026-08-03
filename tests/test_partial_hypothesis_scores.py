from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput

from d5p4.config import Config
from d5p4.dream_ref.modeling_dream import DreamModel
from d5p4.exps.correlation import partial_hypothesis_scores as experiment
from d5p4.exps.correlation.partial_hypothesis_scores import (
    ExperimentSettings,
    ReferenceItem,
    build_diagnostic_table,
    build_mask_draws,
    certainty_scores_from_logits,
    compute_conditional_llama_ppl_batch,
    config_from_remaining_args,
    load_reference_items,
    load_resumable_points,
    parse_experiment_settings,
    score_dream_mask_batch,
    score_llada_mask_batch,
    select_shared_eligible_items,
    summarize_correlations,
)
from d5p4.llada_ref.modeling_llada import LLaDAModelLM


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    chat_template = "template"

    def __call__(self, text: str, *, add_special_tokens: bool, **_: Any) -> dict[str, list[int]]:
        ids = [10 + (ord(char) % 40) for char in text]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        return {"input_ids": ids}

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return f"<user>{messages[0]['content']}<assistant>"


def _item(index: int = 0) -> ReferenceItem:
    return ReferenceItem(
        dataset="truthful_qa",
        task_family="qa",
        item_id=f"truthful_qa:{index}",
        dataset_index=index,
        prompt_text="Q?",
        completion_text="answer",
    )


def test_exact_mask_draws_are_deterministic_and_completion_only():
    input_ids = torch.arange(10)
    kwargs = {
        "prompt_length": 4,
        "mask_token_id": 99,
        "mask_ratio": 0.5,
        "num_draws": 16,
        "seed_parts": (42, "arc", "item", "llada", 0.5),
    }

    masked_a, masks_a, count_a = build_mask_draws(input_ids, **kwargs)
    masked_b, masks_b, count_b = build_mask_draws(input_ids, **kwargs)

    assert count_a == count_b == 3
    assert torch.equal(masked_a, masked_b)
    assert torch.equal(masks_a, masks_b)
    assert torch.equal(masked_a[:, :4], input_ids[:4].expand(16, -1))
    assert torch.all(masks_a.sum(dim=1) == 3)
    assert torch.all((masked_a[:, 4:] == 99) == masks_a)
    assert torch.all((masked_a[:, 4:] != 99).any(dim=1))


def test_certainty_formulas_are_unnormalized_and_row_independent():
    uniform_logits = torch.zeros(1, 2, 4)
    peaked_logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    uniform_entropy, uniform_self = certainty_scores_from_logits(uniform_logits, mask)
    peaked_entropy, peaked_self = certainty_scores_from_logits(peaked_logits, mask)
    combined_entropy, combined_self = certainty_scores_from_logits(
        torch.cat([uniform_logits, peaked_logits]),
        mask.expand(2, -1),
    )

    assert uniform_entropy.item() == pytest.approx(0.0, abs=1e-7)
    assert uniform_self.item() == pytest.approx(0.0, abs=1e-7)
    assert peaked_entropy.item() > uniform_entropy.item()
    assert peaked_self.item() > peaked_entropy.item()
    assert combined_entropy.tolist() == pytest.approx([uniform_entropy.item(), peaked_entropy.item()])
    assert combined_self.tolist() == pytest.approx([uniform_self.item(), peaked_self.item()])


def test_llada_and_dream_alignment_use_expected_logit_positions():
    completion_masks = torch.tensor([[True, False], [False, True]])
    llada_logits = torch.tensor(
        [
            [[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ],
    )
    seen_llada = {}

    class _LLaDA:
        def __call__(self, input_ids, **kwargs):
            seen_llada.update(kwargs)
            return CausalLMOutputWithPast(logits=cast(torch.FloatTensor, llada_logits))

    masked_ids = torch.tensor([[1, 9, 9], [1, 9, 9]])
    llada_scores = score_llada_mask_batch(
        cast(LLaDAModelLM, _LLaDA()),
        masked_ids,
        prompt_length=1,
        completion_length=2,
    )
    expected = certainty_scores_from_logits(llada_logits, torch.ones_like(completion_masks))
    assert torch.equal(llada_scores[0], expected[0])
    assert seen_llada["logits_slice"] == slice(1, None)

    dream_logits = torch.cat([llada_logits, torch.full((2, 1, 3), -20.0)], dim=1)
    seen_dream = {}

    class _Dream:
        def __call__(self, input_ids, **kwargs):
            seen_dream.update(kwargs)
            return MaskedLMOutput(logits=cast(torch.FloatTensor, dream_logits))

    dream_scores = score_dream_mask_batch(
        cast(DreamModel, _Dream()),
        masked_ids,
        completion_length=2,
    )
    assert torch.equal(dream_scores[0], expected[0])
    assert seen_dream["num_logits_to_keep"] == 3
    assert seen_dream["attention_mask"] == "full"


def test_conditional_llama_ppl_scores_only_completion_transitions():
    tokenizer = cast(PreTrainedTokenizerBase, _CharacterTokenizer())
    item = _item()
    seen = {}

    class _PerfectCompletionModel(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, return_dict):
            del return_dict
            seen["attention_mask"] = attention_mask
            vocab_size = 64
            logits = torch.full((*input_ids.shape, vocab_size), -10.0, device=input_ids.device)
            next_tokens = input_ids[:, 1:]
            logits[:, :-1].scatter_(2, next_tokens.unsqueeze(-1), 10.0)
            # Deliberately corrupt early prompt predictions. They must not affect completion NLL.
            logits[:, :2] = 0.0
            return SimpleNamespace(logits=logits)

    result = compute_conditional_llama_ppl_batch(
        _PerfectCompletionModel(),
        tokenizer,
        [item],
        device=torch.device("cpu"),
        use_chat_template=False,
    )

    mean_nll, ppl, token_count = result[0]
    assert token_count == len(item.completion_text)
    assert mean_nll < 1e-5
    assert ppl == pytest.approx(1.0, abs=1e-5)
    assert seen["attention_mask"].sum().item() > token_count


def test_reference_adapters_preserve_full_math_and_canonical_code(monkeypatch):
    math_frame = pd.DataFrame(
        {
            "question": ["Question: 1+1?\nAnswer:"],
            "answer_str": ["Work it out. 1 + 1 = 2. #### 2"],
            "answer_number": ["2"],
        },
    )
    code_frame = pd.DataFrame(
        {
            "task_id": ["HumanEval/0"],
            "prompt": ["def add(a, b):\n"],
            "reference_code": ["    return a + b\n"],
        },
    )
    monkeypatch.setattr(experiment, "gsm8k", lambda _cfg: math_frame)
    monkeypatch.setattr(experiment, "get_code_dataset", lambda _cfg: code_frame)
    config = Config(disable_sys_args=True)

    math_items = load_reference_items(config, "gsm8k")
    code_items = load_reference_items(config, "humaneval")

    assert math_items[0].completion_text == "Work it out. 1 + 1 = 2. #### 2"
    assert code_items[0].completion_text == "    return a + b\n"
    assert code_items[0].item_id == "HumanEval/0"


def test_shared_eligibility_uses_both_tokenizers_and_stable_first_items():
    tokenizers = {
        "llada": cast(PreTrainedTokenizerBase, _CharacterTokenizer()),
        "dream": cast(PreTrainedTokenizerBase, _CharacterTokenizer()),
    }
    source = [_item(index) for index in range(5)]
    config = Config(
        disable_sys_args=True,
        sequence_length=128,
    )

    selected, counts = select_shared_eligible_items(
        source,
        tokenizers=tokenizers,
        config=config,
        num_items=3,
        min_completion_tokens=4,
    )

    assert [item.item_id for item in selected] == ["truthful_qa:0", "truthful_qa:1", "truthful_qa:2"]
    assert counts.source_items == counts.eligible_items == 5


def _point_rows(signature: str = "signature") -> pd.DataFrame:
    rows = []
    for index, ppl in enumerate([1.0, 2.0, 3.0, 4.0]):
        rows.append(
            {
                "experiment_signature": signature,
                "model": "llada",
                "dataset": "truthful_qa",
                "task_family": "qa",
                "item_id": f"item:{index}",
                "mask_ratio": 0.5,
                "mask_ratio_scope": "completion_tokens",
                "score_scope": "all_completion_positions",
                "mask_draws": 16,
                "source_items": 10,
                "eligible_items": 8,
                "entropy_certainty_mean": 5.0 - ppl,
                "self_certainty_mean": ppl,
                "entropy_certainty_sd": 0.1,
                "self_certainty_sd": 0.2,
                "realized_mask_ratio": 0.5,
                "whole_sequence_mask_ratio": 0.25,
                "llama_ppl": ppl,
            },
        )
    return pd.DataFrame(rows)


def test_aggregation_averages_points_and_bootstrap_is_deterministic():
    points = _point_rows()

    first = summarize_correlations(points, bootstrap_samples=100, seed=42)
    second = summarize_correlations(points, bootstrap_samples=100, seed=42)

    pd.testing.assert_frame_equal(first, second)
    row = first.iloc[0]
    assert row["entropy_spearman_rho_vs_ppl"] == pytest.approx(-1.0)
    assert row["entropy_quality_rho"] == pytest.approx(1.0)
    assert row["self_certainty_spearman_rho_vs_ppl"] == pytest.approx(1.0)
    assert row["entropy_quality_advantage"] == pytest.approx(2.0)
    assert row["status"] == "ok"


def test_diagnostic_table_orders_mask_dataset_and_overall_groups():
    base = summarize_correlations(_point_rows(), bootstrap_samples=10, seed=42)
    correlations = pd.concat(
        [
            base.assign(mask_ratio=0.15, dataset="truthful_qa"),
            base.assign(mask_ratio=0.50, dataset="truthful_qa"),
            base.assign(mask_ratio=0.85, dataset="gsm8k", task_family="math"),
        ],
        ignore_index=True,
    )

    table = build_diagnostic_table(correlations)

    assert table[["section", "group"]].to_records(index=False).tolist() == [
        ("masking", "low (15%)"),
        ("masking", "mid (50%)"),
        ("masking", "high (85%)"),
        ("dataset", "TruthfulQA"),
        ("dataset", "GSM8K"),
        ("overall", "all conditions"),
    ]
    overall = table.iloc[-1]
    assert overall["entropy_rho"] == pytest.approx(1.0, abs=2e-6)
    assert overall["self_certainty_rho"] == pytest.approx(-1.0, abs=2e-6)
    assert overall["entropy_minus_self"] == pytest.approx(2.0, abs=4e-6)
    assert overall["entropy_wins"] == "3/3"


def test_resume_rejects_mismatched_signature_and_duplicate_keys(tmp_path):
    path = tmp_path / "points.csv"
    points = _point_rows()
    points.to_csv(path, index=False)

    loaded = load_resumable_points(path, "signature")
    assert len(loaded) == 4
    with pytest.raises(RuntimeError, match="different experiment signature"):
        load_resumable_points(path, "other")

    pd.concat([points, points.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate correlation-point keys"):
        load_resumable_points(path, "signature")


def test_experiment_settings_are_local_and_config_flags_pass_through():
    config = Config(disable_sys_args=True)
    settings, remaining = parse_experiment_settings(
        [
            "--score-correlation-num-items=64",
            "--score-correlation-mask-draws=8",
            "seed=7",
        ],
    )
    parsed_config = config_from_remaining_args(remaining)

    assert not hasattr(config, "score_correlation_num_items")
    assert settings == ExperimentSettings(num_items=64, mask_draws=8)
    assert parsed_config.seed == 7
    assert remaining == ["seed=7"]

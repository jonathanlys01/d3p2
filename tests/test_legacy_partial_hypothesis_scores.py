from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
import torch

from d5p4.exps.correlation.legacy_partial_hypothesis_scores import (
    LEGACY_BATCH_SIZE,
    LEGACY_MC_SAMPLES,
    build_compact_report,
    compute_legacy_internal_scores,
    legacy_mask_batch,
    legacy_scores_from_log_probs,
    parse_legacy_settings,
    summarize,
)
from d5p4.llada_ref.modeling_llada import LLaDAModelLM


def test_legacy_settings_stay_separate_from_config_arguments():
    settings, remaining = parse_legacy_settings(
        [
            "--legacy-datasets=truthful_qa,gsm8k",
            "--legacy-mask-buckets=low,mid,high",
            "--legacy-num-items=12",
            "--legacy-output-prefix=test",
            "--config=experiment.yaml",
        ],
    )

    assert settings.datasets == "truthful_qa,gsm8k"
    assert settings.mask_buckets == "low,mid,high"
    assert settings.num_items == 12
    assert settings.output_prefix == "test"
    assert remaining == ["--config=experiment.yaml"]


def test_legacy_mask_batch_keeps_prompt_and_sweeps_answer_counts():
    torch.manual_seed(42)
    source = torch.arange(10).repeat(LEGACY_BATCH_SIZE, 1)
    masked, counts = legacy_mask_batch(source, prompt_length=2, mask_token_id=99)

    assert torch.equal(masked[:, :2], source[:, :2])
    assert torch.all((masked[:, 2:] == 99).sum(dim=1) == counts)
    assert counts.min().item() >= 1
    assert counts.max().item() <= 8
    assert len(torch.unique(counts)) > 1


@pytest.mark.parametrize(
    ("ratio_range", "minimum", "maximum"),
    [((0.05, 0.25), 1, 5), ((0.40, 0.60), 8, 12), ((0.75, 0.95), 15, 19)],
)
def test_legacy_mask_batch_respects_ratio_bucket(ratio_range, minimum, maximum):
    torch.manual_seed(42)
    source = torch.arange(22).repeat(LEGACY_BATCH_SIZE, 1)
    _, counts = legacy_mask_batch(source, prompt_length=2, mask_ratio_range=ratio_range)

    assert counts.min().item() == minimum
    assert counts.max().item() == maximum


def test_legacy_scores_apply_batch_minmax_to_each_proxy():
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0]],
            [[8.0, 0.0, 0.0]],
        ],
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy, self_certainty = legacy_scores_from_log_probs(log_probs)

    assert entropy.tolist() == pytest.approx([0.0, entropy[1].item(), 1.0])
    assert self_certainty.tolist() == pytest.approx([0.0, self_certainty[1].item(), 1.0])
    assert entropy[1].item() != pytest.approx(self_certainty[1].item())


def test_legacy_internal_score_averages_four_normalized_batches(monkeypatch):
    calls = 0

    def fake_mask(batch, *, prompt_length, mask_token_id=126336, mask_ratio_range=None):
        del prompt_length, mask_token_id, mask_ratio_range
        counts = torch.arange(1, batch.shape[0] + 1)
        return batch, counts

    monkeypatch.setattr(
        "d5p4.exps.correlation.legacy_partial_hypothesis_scores.legacy_mask_batch",
        fake_mask,
    )

    class _Model:
        def __call__(self, input_ids, **kwargs):
            nonlocal calls
            calls += 1
            del kwargs
            batch, length = input_ids.shape
            logits = torch.zeros((batch, length, 3))
            logits[:, :, 0] = torch.arange(batch).unsqueeze(1)
            return SimpleNamespace(logits=logits)

    entropy, self_certainty = compute_legacy_internal_scores(
        cast(LLaDAModelLM, _Model()),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
    )

    assert calls == LEGACY_MC_SAMPLES // LEGACY_BATCH_SIZE == 4
    expected_logits = torch.zeros((LEGACY_BATCH_SIZE, 2, 3))
    expected_logits[:, :, 0] = torch.arange(LEGACY_BATCH_SIZE).unsqueeze(1)
    expected = legacy_scores_from_log_probs(torch.log_softmax(expected_logits, dim=-1))
    assert entropy == pytest.approx(expected[0].mean().item())
    assert self_certainty == pytest.approx(expected[1].mean().item())


def test_legacy_summary_uses_ar_likelihood_higher_is_better():
    points = pd.DataFrame(
        {
            "dataset": ["truthful_qa"] * 4 + ["gsm8k"] * 4,
            "task_family": ["qa"] * 4 + ["math"] * 4,
            "mask_bucket": ["low"] * 8,
            "mask_ratio_low": [0.05] * 8,
            "mask_ratio_high": [0.25] * 8,
            "entropy_score": [0.0, 1.0, 2.0, 3.0] * 2,
            "self_certainty_score": [3.0, 1.0, 2.0, 0.0] * 2,
            "ar_mean_log_likelihood": [0.0, 1.0, 2.0, 3.0] * 2,
        },
    )

    summary = summarize(points)

    assert summary["dataset"].tolist() == ["truthful_qa", "gsm8k"]
    assert summary["entropy_spearman_rho_vs_ar_ll"].tolist() == pytest.approx([1.0, 1.0])
    assert (summary["self_certainty_spearman_rho_vs_ar_ll"] < 0.0).all()
    assert (summary["entropy_advantage"] > 1.0).all()

    compact = build_compact_report(summary)
    assert compact[["group", "value"]].values.tolist() == [
        ["dataset", "truthful_qa"],
        ["dataset", "gsm8k"],
        ["masking", "low [5%, 25%]"],
        ["total", "all conditions"],
    ]
    assert compact["n_conditions"].tolist() == [1, 1, 2, 2]
    assert compact["n_items_per_condition"].tolist() == [4, 4, 4, 4]
    assert (compact["entropy_advantage"] > 1.0).all()

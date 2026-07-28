from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import d5p4.gsm8k_diversity_analysis as diversity_analysis
from d5p4.gsm8k_diversity_analysis import (
    AnalysisCache,
    LexicalMetrics,
    benjamini_hochberg,
    build_prompt_rows,
    candidate_selection_layout,
    classify_bucket,
    compute_lexical_metrics,
    discover_runs,
    final_answer_controlled,
    load_analysis_defaults,
    mean_pairwise_cosine_distance,
    plot_bucket_distributions,
    plot_cross_method_matrix,
    plot_gain,
    plot_recovery_curves,
    run_inference,
    select_candidate_indices,
    validation_summary,
)


def _row(question: str, gold: str, generations: list[str], scores: list[int]) -> dict[str, object]:
    return {
        "question": question,
        "gold_answer": gold,
        "answer_str": f"work #### {gold}",
        "generations": generations,
        "scores": scores,
        "accuracy": sum(scores) / len(scores),
    }


def _write_run(
    root: Path,
    family: str,
    directory: str,
    method: str,
    seed: int,
    rows: list[dict[str, object]],
    **config_overrides: object,
) -> Path:
    path = root / family / directory / f"{method}-{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "llada",
        "qa_dataset": "gsm8k",
        "qa_dataset_len": len(rows),
        "qa_n_shots": 4,
        "method": method,
        "seed": seed,
        "cfg_scale": 2.5,
        "remasking": "selection_temperature",
        "selection_temperature": 0.1,
        "cat_temperature": 1.0,
    }
    config.update(config_overrides)
    path.write_text(json.dumps({"config": config, "results": rows}))
    return path


def _fixture_rows() -> list[dict[str, object]]:
    return [
        _row("q1", "1", ["answer 1", "answer 2"], [1, 0]),
        _row("q2", "2", ["answer 3", "answer 2"], [0, 1]),
        _row("q3", "3", ["answer 4", "answer 5"], [0, 0]),
    ]


def test_analysis_defaults_resolve_server_model_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORK", "/server/work")
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "\n".join(
            (
                "cos_model_id: ${env_path_or:WORK,models/jina/,/fallback/jina/}",
                "cache_dir: ./shared-cache",
            ),
        ),
    )

    defaults = load_analysis_defaults(config_path)

    assert defaults.cos_model_id == "/server/work/models/jina"
    assert defaults.model_cache_dir == (Path.cwd() / "shared-cache").resolve()


def test_lexical_cache_round_trips_nan_as_sqlite_null(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path / "cache.sqlite")
    expected = LexicalMetrics(float("nan"), 0.2, 0.3, 0.5)
    try:
        cache.put_lexical("nan-metric", expected)
        cache.commit()
        actual = cache.get_lexical("nan-metric")
    finally:
        cache.close()

    assert actual is not None
    assert np.isnan(actual.self_bleu)
    assert actual.lexical_diversity == expected.lexical_diversity
    assert actual.pairwise_lexical_distance == expected.pairwise_lexical_distance
    assert actual.unique_fraction == expected.unique_fraction


def test_discovery_matches_by_prompt_not_order_and_supports_seeds(tmp_path: Path) -> None:
    baseline = _fixture_rows()
    method = list(reversed(_fixture_rows()))
    for seed in (0, 1):
        _write_run(tmp_path, "seltemp", "baseline_cfg", "baseline", seed, baseline)
        _write_run(tmp_path, "seltemp", "d5p4_cfg", "greedy_map", seed, method)
    (tmp_path / "seltemp" / "d5p4_cfg" / "temp-checkpoint.json").write_text("{}")
    (tmp_path / "seltemp" / "d5p4_cfg" / "ignored-math-bon-ppl.json").write_text("{}")

    runs = discover_runs(tmp_path)
    summary = validation_summary(runs)

    assert summary["files"] == 4
    assert summary["families"] == 1
    assert summary["replicates"] == 2
    assert summary["prompt_counts"] == [3]
    assert summary["candidate_counts"] == [2]


def test_grouped_methods_select_one_per_group_and_baseline_matches_final_k(tmp_path: Path) -> None:
    generations = [f"candidate {idx}" for idx in range(8)]
    rows = [_row("q", "7", generations, [0, 0, 0, 0, 0, 0, 0, 1])]
    _write_run(tmp_path, "family", "baseline_cfg", "baseline", 0, rows, group_size=1)
    _write_run(tmp_path, "family", "method_cfg", "greedy_map", 0, rows, group_size=2)
    runs = discover_runs(tmp_path)
    layout = candidate_selection_layout(runs)

    assert layout.target_k == 4
    baseline = next(run for run in runs if run.method == "baseline")
    grouped = next(run for run in runs if run.method == "greedy_map")
    baseline_indices = select_candidate_indices(baseline, next(iter(baseline.prompts.values())), 4, 17)
    grouped_indices = select_candidate_indices(grouped, next(iter(grouped.prompts.values())), 4, 17)

    assert len(set(baseline_indices)) == 4
    assert 0 not in baseline_indices
    assert len(grouped_indices) == 4
    assert all(start <= index < start + 2 for start, index in zip(range(0, 8, 2), grouped_indices))
    assert grouped_indices == select_candidate_indices(grouped, next(iter(grouped.prompts.values())), 4, 17)


@pytest.mark.parametrize(
    ("baseline_pass1", "passk", "expected"),
    [
        (1, 1, "Easy"),
        (1, 0, "Easy"),
        (0, 1, "Hard / Recovered"),
        (0, 0, "Unsolved"),
    ],
)
def test_bucket_uses_fixed_baseline_anchor(baseline_pass1: int, passk: int, expected: str) -> None:
    assert classify_bucket(baseline_pass1, passk) == expected


def test_discovery_rejects_missing_baseline_and_unequal_k(tmp_path: Path) -> None:
    _write_run(tmp_path, "family", "method_cfg", "greedy_map", 0, _fixture_rows())
    with pytest.raises(ValueError, match="exactly one"):
        discover_runs(tmp_path)

    other_root = tmp_path / "other"
    _write_run(other_root, "family", "baseline_cfg", "baseline", 0, _fixture_rows())
    unequal = [_row("q1", "1", ["a", "b", "c"], [0, 0, 0]), *_fixture_rows()[1:]]
    _write_run(other_root, "family", "method_cfg", "greedy_map", 0, unequal)
    with pytest.raises(ValueError, match="inconsistent candidate counts"):
        discover_runs(other_root)


def test_discovery_rejects_prompt_mismatch_and_duplicate_method(tmp_path: Path) -> None:
    _write_run(tmp_path, "family", "baseline_cfg", "baseline", 0, _fixture_rows())
    changed = [_row("different", "1", ["a", "b"], [0, 0]), *_fixture_rows()[1:]]
    _write_run(tmp_path, "family", "method_cfg", "greedy_map", 0, changed)
    with pytest.raises(ValueError, match="prompt set differs"):
        discover_runs(tmp_path)

    duplicate_root = tmp_path / "duplicate"
    _write_run(duplicate_root, "family", "baseline_cfg", "baseline", 0, _fixture_rows())
    _write_run(duplicate_root, "family", "method_a", "greedy_map", 0, _fixture_rows())
    _write_run(duplicate_root, "family", "method_b", "greedy_map", 0, _fixture_rows())
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        discover_runs(duplicate_root)


def test_lexical_cosine_and_duplicate_metrics() -> None:
    lexical = compute_lexical_metrics(["same text", "same text"])
    assert lexical.self_bleu == pytest.approx(100.0)
    assert lexical.lexical_diversity == pytest.approx(0.0)
    assert lexical.unique_fraction == pytest.approx(0.5)

    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert mean_pairwise_cosine_distance(embeddings) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    "text",
    [
        "Reasoning here.\n#### 42",
        r"Reasoning here. Therefore \boxed{42}.",
        "Reasoning here.\nThe final answer is 42.",
        "Reasoning here, so the result is 42.",
    ],
)
def test_final_answer_control_reports_masking(text: str) -> None:
    controlled, masked = final_answer_controlled(text)
    assert masked
    assert controlled
    assert controlled != text


def test_prompt_rows_share_baseline_anchor_and_track_incorrect_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_rows = _fixture_rows()
    method_rows = [
        _row("q1", "1", ["method wrong", "method wrong 2"], [0, 0]),
        _row("q2", "2", ["method correct", "method correct 2"], [1, 1]),
        _row("q3", "3", ["method wrong", "method wrong 2"], [0, 0]),
    ]
    _write_run(tmp_path, "family", "baseline_cfg", "baseline", 0, baseline_rows)
    _write_run(tmp_path, "family", "method_cfg", "greedy_map", 0, method_rows)
    runs = discover_runs(tmp_path)

    def fake_embeddings(
        texts: list[str],
        cache: AnalysisCache,
        model_id: str,
        requested_device: str,
        batch_size: int,
        cache_dir: Path,
    ) -> tuple[dict[str, np.ndarray], str]:
        del cache, model_id, requested_device, batch_size, cache_dir
        unique = sorted(set(texts))
        return {
            text: np.asarray([idx + 1.0, 1.0], dtype=np.float32)
            for idx, text in enumerate(unique)
        }, "stub-fingerprint"

    monkeypatch.setattr(diversity_analysis, "load_embeddings", fake_embeddings)
    cache = AnalysisCache(tmp_path / "cache.sqlite")
    try:
        frame, fingerprint = build_prompt_rows(
            runs,
            cache,
            "stub",
            "cpu",
            2,
            1,
            tmp_path / "models",
            7,
        )
    finally:
        cache.close()

    assert fingerprint == "stub-fingerprint"
    for method in ("Standard sampling", "D5P4"):
        method_frame = frame[frame["method_label"] == method].set_index("question")
        assert method_frame.loc["q1", "bucket"] == "Easy"
        assert method_frame.loc["q2", "bucket"] == "Hard / Recovered"
        assert method_frame.loc["q3", "bucket"] == "Unsolved"
    baseline_q1 = frame[(frame["method_label"] == "Standard sampling") & (frame["question"] == "q1")].iloc[0]
    assert baseline_q1["selected_indices"] == "[1]"
    assert baseline_q1["candidate_selection"] == "random_excluding_pass1_anchor"
    assert baseline_q1["incorrect_n"] == 1
    assert np.isnan(baseline_q1["incorrect_semantic_diversity"])


def _analysis_frame() -> pd.DataFrame:
    rows = []
    for prompt_idx in range(16):
        for method_idx, method in enumerate(("Standard sampling", "D5P4")):
            recovered = int((prompt_idx + method_idx) % 3 != 0)
            lexical = 0.1 + 0.03 * recovered + 0.005 * method_idx + 0.001 * prompt_idx
            semantic = 0.2 + 0.04 * recovered + 0.004 * method_idx + 0.001 * prompt_idx
            rows.append(
                {
                    "family": "family",
                    "seed": "0",
                    "method_label": method,
                    "prompt_id": f"p{prompt_idx}",
                    "baseline_pass1": 0,
                    "observed_passk": recovered,
                    "marginal_gain": recovered,
                    "bucket": "Hard / Recovered" if recovered else "Unsolved",
                    "lexical_diversity": lexical,
                    "semantic_diversity": semantic,
                    "unique_fraction": 0.5,
                    "rationale_lexical_diversity": lexical,
                    "rationale_semantic_diversity": semantic,
                    "incorrect_pairwise_lexical_distance": lexical,
                    "incorrect_semantic_diversity": semantic,
                },
            )
    return pd.DataFrame(rows)


def test_inference_is_deterministic_and_bh_is_monotone() -> None:
    frame = _analysis_frame()
    frame["incorrect_semantic_diversity"] = np.nan
    first = run_inference(frame, bootstrap_reps=100, permutation_reps=100, analysis_seed=7)
    second = run_inference(frame, bootstrap_reps=100, permutation_reps=100, analysis_seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["p_adjusted_bh"].dropna() >= first["p_value"].dropna()).all()
    empty_test = first[first["metric"] == "incorrect_semantic_diversity"].iloc[0]
    assert np.isnan(empty_test["effect"])
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_figure_smoke(tmp_path: Path) -> None:
    frame = _analysis_frame()
    easy = frame.head(4).copy()
    easy["baseline_pass1"] = 1
    easy["bucket"] = "Easy"
    easy["marginal_gain"] = 0
    frame = pd.concat([frame, easy], ignore_index=True)
    matrix = pd.DataFrame(
        {
            "method_label": ["Standard sampling", "D5P4"],
            "recovery_rate": [0.5, 0.6],
            "lexical_diversity": [0.1, 0.2],
            "semantic_diversity": [0.2, 0.3],
            "unique_fraction": [0.5, 0.6],
            "recovered_lexical_diversity": [0.12, 0.22],
            "recovered_semantic_diversity": [0.23, 0.33],
        },
    )

    plot_bucket_distributions(frame, tmp_path)
    plot_gain(frame, tmp_path, analysis_seed=4)
    plot_recovery_curves(frame, tmp_path)
    plot_cross_method_matrix(matrix, tmp_path)

    assert len(list(tmp_path.glob("*.png"))) == 5
    assert len(list(tmp_path.glob("*.pdf"))) == 5

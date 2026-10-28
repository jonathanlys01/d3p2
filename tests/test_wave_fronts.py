import itertools
import json
import math

import numpy as np
import pytest

import d5p4.waves.__main__ as cli
from d5p4.waves import fronts, m0
from d5p4.waves.spec import d5p4, diverse_beam, independent, sharded


def test_expected_max_of_subset_matches_brute_force():
    values = [0.0, 1.0, 0.0, 0.3, 1.0, 0.7]
    for k in (1, 2, 4, 6):
        brute = np.mean([max(sub) for sub in itertools.combinations(values, k)])
        assert fronts.expected_max_of_subset(values, k) == pytest.approx(brute)


def test_expected_max_is_unbiased_pass_at_k():
    correct = [1] * 3 + [0] * 13
    assert fronts.expected_max_of_subset(correct, 4) == pytest.approx(1 - math.comb(13, 4) / math.comb(16, 4))


def test_hypervolume_of_a_staircase():
    xs, ys = np.array([70.0, 80.0, 90.0]), np.array([0.8, 0.9, 0.5])
    # dominated region below the corner (100, 0.5): [70,80]x0.3 + [80,100]x0.4; the third point adds nothing
    assert fronts.hypervolume(xs, ys, 100.0, 0.5) == pytest.approx(10 * 0.3 + 20 * 0.4)
    assert fronts.hypervolume(xs, ys, 60.0, 0.5) == 0.0


def test_sharded_runs_share_a_base_tag():
    run = sharded(d5p4(m0.GSM8K, 20.0), 1, 4)
    assert run.tag == "d5p4_w20_t1_s1of4"
    assert run.overrides["qa_num_shards"] == 4
    assert run.overrides["qa_shard_index"] == 1
    assert fronts.base_tag(run.tag) == "d5p4_w20_t1"


def _math_payload(correct_per_question, internal_per_question, indices=None):
    return {
        "text_samples": [
            [f"answer {q} {c} {i}" for i, c in enumerate(row)] for q, row in enumerate(correct_per_question)
        ],
        "internal_scores": internal_per_question,
        "internal_score_metadata": {"higher_is_better": True},
        "results": [{"scores": row} for row in correct_per_question],
        **({"dataset_indices": indices} if indices is not None else {}),
    }


def test_grouped_math_run_picks_one_answer_per_group_by_internal_score():
    run = d5p4(m0.GSM8K, 20.0)
    correct = [[0, 1, 0, 0] * 4]  # group g: candidate 4g + 1 is correct
    internal = [[0.0, 0.1, 0.9, 0.0] * 4]  # but candidate 4g + 2 has the best internal score
    records = fronts.run_question_metrics(run, _math_payload(correct, internal, indices=[7]))

    assert [(r["selection"], r["question"], r["quality"]) for r in records] == [("internal score", 7, 0.0)]


def test_independent_pool_gets_expected_random_and_internal_best_of_n():
    run = independent(m0.GSM8K, 1.0)
    correct = [[1] + [0] * 15]
    internal = [[0.0] + [1.0] * 15]
    records = {r["selection"]: r for r in fronts.run_question_metrics(run, _math_payload(correct, internal))}

    assert records["random k of N"]["quality"] == pytest.approx(4 / 16)
    assert records["random k of N"]["individual"] == pytest.approx(1 / 16)
    assert records["internal score"]["quality"] == 0.0


def test_qa_quality_is_max_f1_against_references():
    run = diverse_beam(m0.ARC_CHALLENGE, 20.0)
    payload = {
        "text_samples": [["the sun", "a dog", "blue sky", "x", "y", "z", "p", "q", "r"]],
        "internal_scores": [[1, 0, 0, 1, 0, 0, 1, 0, 0]],
        "references": [["the sun"]],
    }
    record = fronts.run_question_metrics(run, payload)[0]

    assert record["quality"] == pytest.approx(1.0)
    assert record["individual"] == pytest.approx(1 / 3)


def _records(task, method, tag, quality, self_bleu, selection="internal score"):
    return [
        {
            "task": task,
            "tag": tag,
            "method": method,
            "selection": selection,
            "question": i,
            "quality": q,
            "individual": q,
            "self_bleu": s,
        }
        for i, (q, s) in enumerate(zip(quality, self_bleu, strict=True))
    ]


def test_summarize_reports_paired_differences_and_matched_diversity():
    rng = np.random.default_rng(0)
    n = 400
    hard = rng.random(n) < 0.5  # shared difficulty makes paired intervals tight
    reference_q = (~hard).astype(float)
    better_q = reference_q.copy()
    better_q[np.flatnonzero(hard)[:40]] = 1.0  # +10 points on the same questions
    records = (
        _records("gsm8k", "baseline", "ind_t1", reference_q, np.full(n, 80.0), "random k of N")
        + _records("gsm8k", "baseline", "ind_t1p3", reference_q * 0.9, np.full(n, 70.0), "random k of N")
        + _records("gsm8k", "greedy_map", "d5p4_w10_t1", better_q, np.full(n, 85.0))
        + _records("gsm8k", "greedy_map", "d5p4_w40_t1", better_q, np.full(n, 75.0))
    )
    points, front_rows = fronts.summarize(records, m0.runs(), samples=300)

    by_tag = {(p["selection"], p["tag"]): p for p in points}
    ref = by_tag[("random k of N", "ind_t1")]
    assert ref["diff_vs_ref"] == 0.0
    d5 = by_tag[("internal score", "d5p4_w10_t1")]
    assert d5["diff_vs_ref"] == pytest.approx(10.0)
    assert d5["diff_lo"] > 5.0
    assert d5["param_value"] == 10.0

    front = {(f["method"], f["selection"]): f for f in front_rows}
    assert {f["temperature"] for f in front_rows} == {"", "1"}
    d5p4_front = front[("greedy_map", "internal score")]
    assert d5p4_front["hv_minus_independent"] > 0
    assert d5p4_front["hv_minus_independent_lo"] > 0
    assert d5p4_front["hv_minus_d5p4"] == 0.0
    assert front[("baseline", "random k of N")]["hv_minus_independent"] == 0.0


def test_fronts_command_writes_outputs(tmp_path):
    runs = m0.runs()
    by_tag = {(r.task.name, r.tag): r for r in runs}
    correct = [[1, 0, 0, 0] * 4, [0, 0, 0, 0] * 4]
    internal = [[0.5] * 16, [0.5] * 16]
    for tag in ("ind_t1", "ind_t1p3", "d5p4_w10_t1", "d5p4_w20_t1"):
        run_dir = by_tag[("gsm8k", tag)].results_dir(tmp_path)
        run_dir.mkdir(parents=True)
        (run_dir / "math-x.json").write_text(json.dumps(_math_payload(correct, internal)))

    assert cli.main(["fronts", "--wave", "m0", "--root", str(tmp_path)]) == 0

    out = tmp_path / "fronts"
    assert (out / "per_question.csv").read_text().count("\n") == 1 + 2 * 2 * 2 + 2 * 2
    assert "hypervolume" in (out / "fronts.csv").read_text()
    assert "diff_vs_ref" in (out / "points.csv").read_text()


def test_search_fronts_are_split_by_temperature():
    from d5p4.waves import transfer

    n = 50
    ones, zeros = np.ones(n), np.zeros(n)
    records = (
        _records("truthful_qa", "baseline", "ind_t1", zeros, np.full(n, 50.0), "random k of N")
        + _records("truthful_qa", "greedy_map", "d5p4_w5_t1", ones, np.full(n, 55.0))
        + _records("truthful_qa", "greedy_map", "d5p4_w80_t1", ones, np.full(n, 45.0))
        + _records("truthful_qa", "greedy_map", "d5p4_w5_t1p3", ones, np.full(n, 40.0))
        + _records("truthful_qa", "greedy_map", "d5p4_w80_t1p3", ones, np.full(n, 35.0))
    )
    points, front_rows = fronts.summarize(records, transfer.runs(), samples=50)

    d5p4_fronts = {f["temperature"]: f for f in front_rows if f["method"] == "greedy_map"}
    assert set(d5p4_fronts) == {"1", "1.3"}
    assert d5p4_fronts["1"]["points"] == 2
    assert d5p4_fronts["1.3"]["hv_minus_d5p4"] > 0
    assert {p["temperature"] for p in points if p["tag"].endswith("t1p3")} == {"1.3"}

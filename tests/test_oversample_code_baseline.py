import random

from d5p4._oversample_code_baseline import _selected_code_results, _validation_groups
from d5p4.code_eval import CodeEvaluator


def _validation(passed: bool) -> dict:
    return {
        "extracted_code": "",
        "full_code": "",
        "parse_ok": True,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "error": "",
        "stdout": "",
        "stderr": "",
    }


def test_code_internal_score_selection_reuses_stored_validation_rows():
    rows = [
        {
            "generations": ["low", "high", "mid"],
            "validation": [_validation(False), _validation(True), _validation(False)],
        },
    ]

    selected, indices = _selected_code_results(
        rows=rows,
        metric="int",
        subsample_k=2,
        internal_scores=[[0.1, 0.9, 0.4]],
        random_seed=0,
    )

    assert indices == [[1, 2]]
    assert selected[0]["generations"] == ["high", "mid"]
    assert selected[0]["scores"] == [1, 0]
    assert selected[0]["accuracy"] == 0.5

    metrics = CodeEvaluator().evaluate(_validation_groups(selected))
    assert metrics["accuracy"] == 0.5
    assert metrics["pass@1"] == 0.5


def test_code_random_selection_is_deterministic():
    rows = [
        {
            "generations": ["a", "b", "c", "d"],
            "validation": [_validation(False), _validation(False), _validation(True), _validation(True)],
        },
    ]
    expected = sorted(random.Random(123).sample(range(4), 2))

    _selected, indices = _selected_code_results(
        rows=rows,
        metric="random",
        subsample_k=2,
        internal_scores=None,
        random_seed=123,
    )

    assert indices == [expected]

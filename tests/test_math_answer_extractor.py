"""lm-eval `gsm8k_cot` answer extraction (Dream-Instruct's published protocol) next to the universal parser.

The texts are Dream-v0-Instruct-7B generations from the E2 wave (GSM8K test shard 0, T 1.0).
"""

import pytest

from d5p4.config import Config
from d5p4.eval_core import MathEvaluator
from d5p4.text_postprocessors import LM_EVAL_INVALID, lm_eval_gsm8k_extract, lm_eval_gsm8k_normalize


METAMATH_STYLE = (
    "Ella earns $33 / 3 hours = $11 per hour.\nAfter 12 hours, Ella would earn $11 x 12 = $132.\nThe answer is: 132."
)
WRONG_ARITHMETIC = (
    "Joelle's orchids have 5 * 5 = 25 petals. The daisies have 4 * 10 = 40 petals. "
    "The daisies have 40 - 25 = 20 more petals.\nThe answer is: 20"
)


def test_flexible_extract_takes_the_last_number():
    assert lm_eval_gsm8k_extract(METAMATH_STYLE) == "132."
    assert lm_eval_gsm8k_extract("Samantha has $125\n#### 400\nThe answer is: 400") == "400"
    assert lm_eval_gsm8k_extract("no digits here") == LM_EVAL_INVALID


def test_strict_match_needs_the_exact_cot_phrase():
    assert lm_eval_gsm8k_extract("So 5 + 4 = 9. The answer is 9.", mode="strict") == "9"
    # Dream answers in the MetaMath style "The answer is: N", which the strict regex never matches.
    assert lm_eval_gsm8k_extract(METAMATH_STYLE, mode="strict") == LM_EVAL_INVALID


def test_normalisation_follows_regexes_to_ignore():
    assert lm_eval_gsm8k_normalize("$1,234.") == "1234"
    assert lm_eval_gsm8k_normalize("steps\n#### 72") == "72"
    # exact_match does not parse numbers: "26.00" stays wrong against "26", as in lm-eval.
    assert lm_eval_gsm8k_normalize("$26.00") == "26.00"


@pytest.mark.parametrize(
    ("text", "gold", "universal", "lm_eval"),
    [
        # The universal parser reads the "$33 / 3 hours = $" span as inline LaTeX and scores 11.
        (METAMATH_STYLE, "132", 0, 1),
        # It evaluates the last arithmetic expression (40 - 25 = 15) instead of reading the stated answer (20).
        (WRONG_ARITHMETIC, "15", 1, 0),
        ("Carmen has $100, Samantha has $125, and Daisy has $175. Combined, they have $400.", "400", 1, 1),
        ("55", "75", 0, 0),
    ],
)
def test_extractors_on_dream_generations(text, gold, universal, lm_eval):
    assert MathEvaluator().check(text, gold) == universal
    assert MathEvaluator(answer_extractor="lm_eval_flexible").check(text, gold) == lm_eval


def test_evaluate_uses_the_chosen_extractor_in_workers():
    generations = [[METAMATH_STYLE, "The answer is: 7"], [WRONG_ARITHMETIC]]
    golds = ["132", "15"]

    lm_eval = MathEvaluator(answer_extractor="lm_eval_flexible").evaluate(generations, golds, num_workers=2)
    universal = MathEvaluator().evaluate(generations, golds, num_workers=2)

    assert lm_eval["pass@1"] == pytest.approx(0.25)
    assert universal["pass@1"] == pytest.approx(0.5)


def test_config_rejects_unknown_extractors():
    with pytest.raises(AssertionError):
        Config(disable_sys_args=True, math_answer_extractor="regex")
    with pytest.raises(ValueError, match="answer_extractor"):
        MathEvaluator(answer_extractor="regex")

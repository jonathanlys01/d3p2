"""Transfer check: the frozen front grid on the three datasets M0 did not cover.

TruthfulQA, CommonsenseQA and HumanEval, each on its full reported split, with the 12-run front of
`d5p4.waves.spec.front_runs`. Nothing is selected from these splits: the grid comes from M0 and each front is reported
whole. These runs are also the three datasets' fronts in the draft (P1) of the D5P4 Pareto plan.

Same LLaDA profile as M0 (no CFG, pure diffusion, seed 0, fixed question order). The most expensive task comes first,
so its long runs start first.

Runs 36 onward add Diverse Beam Search and D5P4 at T 1.3 on TruthfulQA, where independent sampling at T 1.3 beat every
search point at T 1.0: they test whether search on top of temperature still helps. They are appended so the original
array indices stay valid.
"""

from __future__ import annotations

from d5p4.waves.m0 import LLADA_NO_CFG
from d5p4.waves.spec import (
    FRONT_DIVERSITY_ALPHAS,
    FRONT_W_INTERACTIONS,
    Run,
    Task,
    check_unique_tags,
    d5p4,
    diverse_beam,
    front_runs,
    generation_length,
)


EVAL_SPLIT = {**LLADA_NO_CFG, "dataset_split": "eval"}

HUMANEVAL = Task(
    name="humaneval",
    entry="llada_code.py",
    family="code",
    limit_key="code_dataset_len",
    n_groups=3,
    group_size=3,
    overrides={
        **EVAL_SPLIT,
        **generation_length(512),
        "code_dataset": "humaneval",
        "code_dataset_len": -1,
        "code_n_shots": 0,
        "confidence_eos_eot_inf": True,
        "skip_eval": False,
    },
)


def _qa_task(name: str) -> Task:
    return Task(
        name=name,
        entry="single_run_llada.py",
        family="qa",
        limit_key="qa_dataset_len",
        n_groups=3,
        group_size=3,
        overrides={
            **EVAL_SPLIT,
            **generation_length(128),
            "qa_dataset": name,
            "qa_dataset_len": -1,
            "qa_n_shots": 0,
            "confidence_eos_eot_inf": False,
            "skip_eval": True,
        },
    )


COMMONSENSE_QA = _qa_task("commonsense_qa")
TRUTHFUL_QA = _qa_task("truthful_qa")

TASKS = (HUMANEVAL, COMMONSENSE_QA, TRUTHFUL_QA)
HOT_TEMPERATURE = 1.3


def runs() -> list[Run]:
    result = [run for task in TASKS for run in front_runs(task)]
    result.extend(diverse_beam(TRUTHFUL_QA, alpha, HOT_TEMPERATURE) for alpha in FRONT_DIVERSITY_ALPHAS)
    result.extend(d5p4(TRUTHFUL_QA, w, HOT_TEMPERATURE) for w in FRONT_W_INTERACTIONS)
    return check_unique_tags(result)

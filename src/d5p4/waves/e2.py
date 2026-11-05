"""E2: Dream-v0-Instruct-7B, a second diffusion language model, on GSM8K and TruthfulQA.

Dream's own sampler (entropy-ordered unmasking, top-p 0.9), with the frozen grid. Its D5P4 kernel uses Dream's hidden
states, so the alpha / beta scale may not carry over from LLaDA: `dream_gsm8k_dev` checks on 200 GSM8K train questions
that the grid spans Dream's diversity range. Its results only describe the grid; nothing is selected from them. The
reported fronts use the same questions as P1 and the transfer wave (GSM8K test shard 0 of 4, TruthfulQA validation).
Two seeds; seed 0 of every run comes first.
"""

from __future__ import annotations

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
    independent,
    seeded,
    sharded,
    variant,
)


SEEDS = (0, 1)
GSM8K_SHARDS = 4

DREAM = {
    "model": "dream",
    "seed": 0,
    "dataset_shuffle_seed": 0,
    "compile_model": True,
    "dream_eps": 1e-3,
    "dream_alg": "entropy",
    "dream_alg_temp": 0.0,
    "dream_top_p": 0.9,
    "transversal": True,
    "subsample_start": 0,
    "interactive": False,
    "minimal_log": True,
    "standalone_job": True,
    "resume_runs": True,
    "resume_db_keep_completed": True,
}


def dream_length(tokens: int) -> dict[str, int]:
    return {**generation_length(tokens), "dream_steps": tokens}


GSM8K_DEV = Task(
    name="dream_gsm8k_dev",
    entry="dream_math.py",
    family="math",
    limit_key="qa_dataset_len",
    n_groups=4,
    group_size=4,
    overrides={
        **DREAM,
        **dream_length(256),
        "dataset_split": "dev",
        "qa_dataset": "gsm8k",
        "qa_dataset_len": 200,
        "qa_n_shots": 0,
        "skip_eval": False,
    },
)
GSM8K = variant(GSM8K_DEV, "dream_gsm8k", dataset_split="eval", qa_dataset_len=-1)
TRUTHFUL_QA = Task(
    name="dream_truthful_qa",
    entry="single_run_dream.py",
    family="qa",
    limit_key="qa_dataset_len",
    n_groups=3,
    group_size=3,
    overrides={
        **DREAM,
        **dream_length(128),
        "dataset_split": "eval",
        "qa_dataset": "truthful_qa",
        "qa_dataset_len": -1,
        "qa_n_shots": 0,
        "skip_eval": True,
    },
)


def grid_check_runs(task: Task) -> list[Run]:
    return [
        independent(task, 1.0),
        *(diverse_beam(task, alpha) for alpha in FRONT_DIVERSITY_ALPHAS),
        *(d5p4(task, w) for w in FRONT_W_INTERACTIONS),
    ]


def runs() -> list[Run]:
    result: list[Run] = []
    for seed in SEEDS:
        result.extend(seeded(run, seed) for run in grid_check_runs(GSM8K_DEV))
        result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in front_runs(GSM8K))
        result.extend(seeded(run, seed) for run in front_runs(TRUTHFUL_QA))
    return check_unique_tags(result)

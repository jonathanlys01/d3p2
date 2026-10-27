"""M0: dev-split hyperparameter sweeps for LLaDA without classifier-free guidance.

Each method's diversity parameter is swept on a split disjoint from the reported benchmark split, so every
method's operating point for the main tables is picked by the same rule:

- GSM8K: 500 questions from the train split (4 groups x 4, independent pool of 16).
- ARC-Challenge: the 299-question validation split (3 x 3, pool of 9). TruthfulQA has no disjoint split.
- MBPP: the full config's 90-task validation split, 4-shot (3 x 3, pool of 9). HumanEval has no disjoint split.

One seed. The dev questions are fixed by `dataset_shuffle_seed` so later seeds see the same questions.
"""

from __future__ import annotations

from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, diverse_beam, greedy_beam, independent


W_INTERACTIONS = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
DIVERSITY_ALPHAS = (5.0, 10.0, 20.0, 40.0, 80.0)
INDEPENDENT_TEMPERATURES = (0.7, 1.0, 1.3)

# LLaDA-8B-Instruct, pure diffusion (one block), one token unmasked per step, no guidance in the original LLaDA
# convention (cfg_scale=0 with llada_reference_cfg=true skips the unconditional pass).
LLADA_NO_CFG = {
    "model": "llada",
    "seed": 0,
    "dataset_split": "dev",
    "dataset_shuffle_seed": 0,
    "compile_model": True,
    "remasking": "low_confidence",
    "selection_temperature": 0.0,
    "logits_eos_inf": False,
    "cfg_scale": 0,
    "llada_reference_cfg": True,
    "transversal": True,
    "subsample_start": 0,
    "interactive": False,
    "minimal_log": True,
    "standalone_job": True,
    "resume_runs": True,
    "resume_db_keep_completed": True,
}


def _length(tokens: int) -> dict[str, int]:
    return {"gen_length": tokens, "llada_steps": tokens, "block_length": tokens, "subsample_end": tokens}


GSM8K = Task(
    name="gsm8k",
    entry="llada_math.py",
    family="math",
    limit_key="qa_dataset_len",
    n_groups=4,
    group_size=4,
    overrides={
        **LLADA_NO_CFG,
        **_length(256),
        "qa_dataset": "gsm8k",
        "qa_dataset_len": 500,
        "qa_n_shots": 0,
        "confidence_eos_eot_inf": False,
        "skip_eval": False,
    },
)

ARC_CHALLENGE = Task(
    name="ai2_arc",
    entry="single_run_llada.py",
    family="qa",
    limit_key="qa_dataset_len",
    n_groups=3,
    group_size=3,
    overrides={
        **LLADA_NO_CFG,
        **_length(128),
        "qa_dataset": "ai2_arc",
        "qa_dataset_len": -1,
        "qa_n_shots": 0,
        "confidence_eos_eot_inf": False,
        "skip_eval": True,
    },
)

MBPP = Task(
    name="mbpp",
    entry="llada_code.py",
    family="code",
    limit_key="code_dataset_len",
    n_groups=3,
    group_size=3,
    overrides={
        **LLADA_NO_CFG,
        **_length(256),
        "code_dataset": "mbpp",
        "code_dataset_len": -1,
        "code_n_shots": 4,
        "confidence_eos_eot_inf": True,
        "skip_eval": False,
    },
)

TASKS = (GSM8K, ARC_CHALLENGE, MBPP)


def runs() -> list[Run]:
    result: list[Run] = []
    for task in TASKS:
        result.extend(independent(task, temperature) for temperature in INDEPENDENT_TEMPERATURES)
        result.append(greedy_beam(task))
        result.extend(diverse_beam(task, alpha) for alpha in DIVERSITY_ALPHAS)
        result.extend(d5p4(task, w) for w in W_INTERACTIONS)
    return check_unique_tags(result)

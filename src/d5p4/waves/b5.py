"""B5: Dream sanity check (`b5`), then the redone Dream wave under Dream-Instruct's own protocol (`b5_dream`).

E2 ran Dream-v0-Instruct-7B with Dream's sampler (entropy-ordered unmasking, top-p 0.9, 256 tokens one per step) at
the LLaDA front temperatures and reached about 29% per answer on GSM8K. Dream's published 81.0 comes from lm-eval
`gsm8k_cot`, zero-shot, chat template, 256 tokens / 256 steps, entropy, top-p 0.9 and temperature 0.1, scored by
flexible-extract (HKUNLP/Dream eval_instruct/eval.sh). E2's sampler matches Dream's `diffusion_generate` token for
token (tests/test_dream_reference_parity.py); the suspected gap is the temperature. At T 0.7 and 1.0, a third to a
half of E2's answers are a bare number with no reasoning: the end-of-text run is committed before the reasoning. With
top-p 0.9 at T 0.1, most positions keep a single candidate, their entropies tie at zero and the leftmost is unmasked
first, so decoding runs nearly left to right. E2 also had "\\n\\n" instead of "\\n" before "A:" and was scored with the
universal parser instead of flexible-extract; both are fixed here.

`b5` (check, about 2.5 H100-h): indices 0-4 decode the first 50 questions of E2's GSM8K shard 0 (`qa_dataset_len`
200, shard 0 of 4) greedily and at T 0.1, 0.3, 0.5 and 1.0; index 5 decodes all of HumanEval once at T 0.1 with
Dream's `humaneval_instruct` protocol (768 tokens / 768 steps), to compare with the published 55.5. `b5_dream` is the
front to run if the check reproduces Dream's numbers: GSM8K shard 0 as E2, two seeds, and HumanEval, one seed. Search
runs use the official T 0.1; independent sampling spans `IND_TEMPERATURES`, to be set from the check (the highest
temperatures that do not collapse).
"""

from __future__ import annotations

from dataclasses import replace

from d5p4.waves.e2 import DREAM, GSM8K_SHARDS, dream_length
from d5p4.waves.spec import (
    FRONT_DIVERSITY_ALPHAS,
    FRONT_W_INTERACTIONS,
    Run,
    Task,
    check_unique_tags,
    d5p4,
    diverse_beam,
    greedy_beam,
    independent,
    seeded,
    sharded,
)


OFFICIAL_TEMPERATURE = 0.1
CHECK_TEMPERATURES = (0.1, 0.3, 0.5, 1.0)
IND_TEMPERATURES = (0.1, 0.3, 0.5)
SEEDS = (0, 1)

DREAM_OFFICIAL = {**DREAM, "math_answer_extractor": "lm_eval_flexible"}

GSM8K_CHECK = Task(
    name="dream_gsm8k_check",
    entry="dream_math.py",
    family="math",
    limit_key="qa_dataset_len",
    n_groups=4,
    group_size=1,
    overrides={
        **DREAM_OFFICIAL,
        **dream_length(256),
        "dataset_split": "eval",
        "qa_dataset": "gsm8k",
        "qa_dataset_len": 200,  # shard 0 of 4 of the first 200: the first 50 questions of E2's shard 0
        "qa_n_shots": 0,
        "skip_eval": False,
    },
)
GSM8K_CHECK_GREEDY = replace(GSM8K_CHECK, name="dream_gsm8k_check_greedy", n_groups=1)

HUMANEVAL = Task(
    name="dream_humaneval",
    entry="dream_code.py",
    family="code",
    limit_key="code_dataset_len",
    n_groups=3,
    group_size=3,
    overrides={
        **DREAM_OFFICIAL,
        **dream_length(768),
        "dataset_split": "eval",
        "code_dataset": "humaneval",
        "code_dataset_len": -1,
        "code_n_shots": 0,
        "skip_eval": False,
    },
    reference_temperature=OFFICIAL_TEMPERATURE,
)
HUMANEVAL_CHECK = replace(HUMANEVAL, name="dream_humaneval_check", n_groups=1, group_size=1)

GSM8K = replace(
    GSM8K_CHECK,
    name="dream_gsm8k_official",
    n_groups=4,
    group_size=4,
    overrides={**GSM8K_CHECK.overrides, "qa_dataset_len": -1},
    reference_temperature=OFFICIAL_TEMPERATURE,
)


def dream_front(task: Task) -> list[Run]:
    """The 12-run front with Dream's temperature: search at T 0.1, independent sampling over `IND_TEMPERATURES`."""
    return [
        *(independent(task, temperature) for temperature in IND_TEMPERATURES),
        greedy_beam(task, OFFICIAL_TEMPERATURE),
        *(diverse_beam(task, alpha, OFFICIAL_TEMPERATURE) for alpha in FRONT_DIVERSITY_ALPHAS),
        *(d5p4(task, w, OFFICIAL_TEMPERATURE) for w in FRONT_W_INTERACTIONS),
    ]


def runs() -> list[Run]:
    """The sanity check: four short GSM8K runs (indices 0-3), then one pass over HumanEval (index 4)."""
    return check_unique_tags(
        [
            sharded(independent(GSM8K_CHECK_GREEDY, 0.0), 0, GSM8K_SHARDS),
            *(sharded(independent(GSM8K_CHECK, t), 0, GSM8K_SHARDS) for t in CHECK_TEMPERATURES),
            independent(HUMANEVAL_CHECK, OFFICIAL_TEMPERATURE),
        ],
    )


def dream_runs() -> list[Run]:
    """HumanEval first (its runs are the longest), then GSM8K shard 0 with seed 0 before seed 1."""
    result = [seeded(run, 0) for run in dream_front(HUMANEVAL)]
    for seed in SEEDS:
        result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in dream_front(GSM8K))
    return check_unique_tags(result)

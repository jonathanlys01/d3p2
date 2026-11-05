"""S1: classifier-free guidance collapse on GSM8K and TruthfulQA.

Guidance sharpens LLaDA's predictions and collapses the diversity of independent samples; the question is whether
search keeps coverage (pass@k, max F1) as guidance grows. Levels follow the paper's convention, where 1.0 is no
guidance and the earlier tables used 2.0 (QA) and 2.5 (GSM8K); the no-guidance level is the P1 / transfer front.

Each level is its own task (`gsm8k_cfg2`, ...) with its own independent-sampling reference, on the same questions as
P1 and the transfer wave: GSM8K test shard 0 of 4 and the full TruthfulQA validation split. Per level: independent
sampling at T 1.0 and 1.3, Greedy Beam Search, and Diverse Beam Search and D5P4 at the low and high ends of the
frozen grid. Two seeds; seed 0 of every run comes first.
"""

from __future__ import annotations

from d5p4.waves import p1, transfer
from d5p4.waves.spec import (
    Run,
    Task,
    check_unique_tags,
    d5p4,
    diverse_beam,
    greedy_beam,
    independent,
    seeded,
    sharded,
    value_tag,
    variant,
)


CFG_LEVELS = (1.5, 2.0, 2.5, 3.0)
SEEDS = (0, 1)
GSM8K_SHARDS = 4


def reference_scale(level: float) -> float:
    """`cfg_scale` for a paper-convention level: with `llada_reference_cfg`, logits scale by `cfg_scale + 1`."""
    return level - 1.0


def level_task(task: Task, level: float) -> Task:
    return variant(task, f"{task.name}_cfg{value_tag(level)}", cfg_scale=reference_scale(level))


def level_runs(task: Task) -> list[Run]:
    return [
        independent(task, 1.0),
        independent(task, 1.3),
        greedy_beam(task),
        diverse_beam(task, 20.0),
        diverse_beam(task, 320.0),
        d5p4(task, 20.0),
        d5p4(task, 320.0),
    ]


def runs() -> list[Run]:
    result: list[Run] = []
    for seed in SEEDS:
        for level in CFG_LEVELS:
            task = level_task(p1.GSM8K, level)
            result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in level_runs(task))
        for level in CFG_LEVELS:
            result.extend(seeded(run, seed) for run in level_runs(level_task(transfer.TRUTHFUL_QA, level)))
    return check_unique_tags(result)

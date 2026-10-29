"""P1: draft quality-diversity fronts on the reported splits of GSM8K, ARC-Challenge and MBPP.

The frozen 12-run front of `d5p4.waves.spec.front_runs` at T 1.0, plus Diverse Beam Search and D5P4 at T 1.3 over the
same alpha / beta grid: on TruthfulQA, search at T 1.3 beat independent sampling at T 1.3 by about 4 max F1 at matched
Self-BLEU, while every T 1.0 search point lost to it. The other three datasets get their T 1.3 runs in the transfer
wave.

Same LLaDA profile and pools as M0 (no CFG, pure diffusion, seed 0, fixed question order), on the evaluation splits:

- GSM8K: question shard 0 of 4 of the test split (330 of 1,319), 4 x 4. Shards 1-3 extend it later without
  regenerating anything.
- MBPP: the 257-task sanitized test split, 4-shot from the sanitized train split, 3 x 3.
- ARC-Challenge: the 1,172-question test split, 3 x 3.

Nothing is selected on these splits: the grid comes from M0 and each front is reported whole. The most expensive task
comes first, so its long runs start first.
"""

from __future__ import annotations

from d5p4.waves import m0
from d5p4.waves.spec import Run, Task, check_unique_tags, front_runs, search_runs, sharded


HOT_TEMPERATURE = 1.3
GSM8K_SHARDS = 4


def _eval_split(task: Task, **overrides: object) -> Task:
    return Task(
        name=task.name,
        entry=task.entry,
        family=task.family,
        limit_key=task.limit_key,
        n_groups=task.n_groups,
        group_size=task.group_size,
        overrides={**task.overrides, "dataset_split": "eval", **overrides},
    )


GSM8K = _eval_split(m0.GSM8K, qa_dataset_len=-1)
MBPP = _eval_split(m0.MBPP)
ARC_CHALLENGE = _eval_split(m0.ARC_CHALLENGE)

TASKS = (GSM8K, MBPP, ARC_CHALLENGE)


def _task_runs(task: Task) -> list[Run]:
    return [*front_runs(task), *search_runs(task, HOT_TEMPERATURE)]


def runs() -> list[Run]:
    result = [sharded(run, 0, GSM8K_SHARDS) for run in _task_runs(GSM8K)]
    result.extend(run for task in TASKS[1:] for run in _task_runs(task))
    return check_unique_tags(result)

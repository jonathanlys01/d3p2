"""B7 (optional): the k x w scale pilot on GSM8K, the second task asked for by the reviewers.

Same design as `d5p4.waves.scale` (TruthfulQA): D5P4 beta 20, DBS alpha 80 and independent sampling at T 1.0 with
the same pool, at 8 x 8, 16 x 4 and 4 x 8. The 4 x 4 row is P1 on the same questions (test shard 0 of 4, 330
questions, seed 0). pass@k saturates on GSM8K as k grows (k = 8, 16 here), which is why TruthfulQA went first.

Cost is linear in the pool (measured on the TruthfulQA pilot from 9 to 128 sequences): GSM8K at 4 x 4 takes 3.5 h
per run, so about 7 h at pool 32 and 14 h at pool 64 (qos_gpu_h100-t3 allows 20 h). Memory at pool 64 x 256 block
positions is that of the TruthfulQA pool 128 x 128 runs (52 GB peak). The heaviest pools come first.
"""

from __future__ import annotations

from dataclasses import replace

from d5p4.waves import p1
from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, diverse_beam, independent, seeded, sharded


SHAPES = ((8, 8), (16, 4), (4, 8))
DIVERSITY_ALPHA = 80.0
W_INTERACTION = 20.0
SHARDS = 4
SEED = 0


def shape_task(n_groups: int, group_size: int) -> Task:
    return replace(p1.GSM8K, name=f"gsm8k_k{n_groups}w{group_size}", n_groups=n_groups, group_size=group_size)


TASKS = tuple(shape_task(k, w) for k, w in SHAPES)


def shape_runs(task: Task) -> list[Run]:
    return [d5p4(task, W_INTERACTION), diverse_beam(task, DIVERSITY_ALPHA), independent(task, 1.0)]


def runs() -> list[Run]:
    return check_unique_tags([sharded(seeded(run, SEED), 0, SHARDS) for task in TASKS for run in shape_runs(task)])

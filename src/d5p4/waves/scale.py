"""Scale pilot: D5P4 against Diverse Beam Search as the number of groups k and the group size w grow.

The synthetic study of the paper (appendix "Scaling Behavior") finds Diverse Beam Search competitive only at low
combinatorial complexity k log w, and D5P4 ahead as it grows. Every text run so far sits at k log w <= 5.5 (3 x 3 and
4 x 4), where both tie. This pilot runs larger k x w on TruthfulQA (max F1 does not saturate at large k the way pass@k
does on GSM8K), at the pre-specified points DBS alpha 80 and D5P4 beta 20, next to independent sampling with the same
pool.

Cost grows with the pool k * w: generation is compute-bound (CFG, which doubles the batch, took 1.84x as long), so a
run costs about 5.5 s per question per 9 sequences in the pool. Pools of 128 also approach the memory limit of one
H100, since every selection step holds several pool x 128 x 126k full-vocabulary tensors. Probe first:

    SMOKE_LIMIT=5 .jz_waves/submit.sh --smoke --no-eval --max-parallel 8 scale

then read the per-question times (`generated_at` in the resume databases) and the GPU memory logs before running the
shard. Question shard 0 of 4 (205 questions), seed 0. The heaviest pools come first so their long runs start first.
"""

from __future__ import annotations

from dataclasses import replace

from d5p4.waves import transfer
from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, diverse_beam, independent, seeded, sharded


# (k groups, w per group), heaviest pool first: 128 as 8 x 16 and 16 x 8, 64 as 8 x 8 and 16 x 4, then 4 x 8.
SHAPES = ((16, 8), (8, 16), (8, 8), (16, 4), (4, 8))
DIVERSITY_ALPHA = 80.0
W_INTERACTION = 20.0
SHARDS = 4
SEED = 0


def shape_task(n_groups: int, group_size: int) -> Task:
    return replace(
        transfer.TRUTHFUL_QA,
        name=f"truthful_qa_k{n_groups}w{group_size}",
        n_groups=n_groups,
        group_size=group_size,
    )


TASKS = tuple(shape_task(k, w) for k, w in SHAPES)


def shape_runs(task: Task) -> list[Run]:
    return [d5p4(task, W_INTERACTION), diverse_beam(task, DIVERSITY_ALPHA), independent(task, 1.0)]


def runs() -> list[Run]:
    return check_unique_tags(
        [sharded(seeded(run, SEED), 0, SHARDS) for task in TASKS for run in shape_runs(task)],
    )

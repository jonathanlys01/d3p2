"""Parallel decoding (ARR B1): several tokens unmasked per denoising step instead of one.

Every LLaDA run so far uses as many steps as generated tokens. Here the same generation length is decoded in half and
a quarter of the steps (2 and 4 tokens per step): GSM8K 256 tokens in 128 and 64 steps, TruthfulQA 128 tokens in 64
and 32 steps. Pure diffusion is kept (one block), so only the step count changes. Selection runs at every step, as
before: the subsample window is indexed by step and `generation_length` sets its end to the token count, past the
last step. Each step, every child commits the `gen_length / steps` most confident of its own sampled tokens, then
the selector picks among the children; nothing in the selection depends on how many tokens a step commits.

Methods at the pre-specified points: independent sampling at T 1.0 (scored as random k of N and by internal score),
DBS alpha 80 and D5P4 beta 20. Questions: GSM8K test shard 0 of 4 (330) and TruthfulQA shard 0 of 4 (205), seed 0.
The one-token-per-step references are P1 (`p1/results/gsm8k/*_t1_s0of4`) and the transfer wave's full-split
TruthfulQA runs restricted to shard 0. Each task variant has its own independent run, so the fronts summary works.
"""

from __future__ import annotations

from d5p4.waves import p1, transfer
from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, diverse_beam, independent, sharded, variant


TOKENS_PER_STEP = (2, 4)
DIVERSITY_ALPHA = 80.0
W_INTERACTION = 20.0
SHARDS = 4


def parallel_task(task: Task, tokens_per_step: int) -> Task:
    gen_length = task.overrides["gen_length"]
    assert gen_length % tokens_per_step == 0
    return variant(task, f"{task.name}_tps{tokens_per_step}", llada_steps=gen_length // tokens_per_step)


TASKS = tuple(parallel_task(task, n) for task in (p1.GSM8K, transfer.TRUTHFUL_QA) for n in TOKENS_PER_STEP)


def task_runs(task: Task) -> list[Run]:
    return [independent(task, 1.0), diverse_beam(task, DIVERSITY_ALPHA), d5p4(task, W_INTERACTION)]


def runs() -> list[Run]:
    return check_unique_tags([sharded(run, 0, SHARDS) for task in TASKS for run in task_runs(task)])

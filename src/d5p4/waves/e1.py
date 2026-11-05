"""E1: semi-autoregressive LLaDA (blocks of 32 tokens) on GSM8K and MBPP.

LLaDA's reported math and code results decode in blocks; the main results use pure diffusion (one block). Same
questions, step count and frozen 12-run front as P1 (GSM8K test shard 0 of 4, MBPP sanitized test), with the
answer generated as 8 blocks of 32 tokens, 32 steps each. Two seeds; seed 0 of every run comes first.
"""

from __future__ import annotations

from d5p4.waves import p1
from d5p4.waves.spec import Run, check_unique_tags, front_runs, seeded, sharded, variant


BLOCK_LENGTH = 32
SEEDS = (0, 1)
GSM8K_SHARDS = 4

GSM8K = variant(p1.GSM8K, "gsm8k_semiar", block_length=BLOCK_LENGTH)
MBPP = variant(p1.MBPP, "mbpp_semiar", block_length=BLOCK_LENGTH)


def runs() -> list[Run]:
    result: list[Run] = []
    for seed in SEEDS:
        result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in front_runs(GSM8K))
        result.extend(seeded(run, seed) for run in front_runs(MBPP))
    return check_unique_tags(result)

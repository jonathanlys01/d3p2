"""B6: is D5P4's edge over DBS under tempered remasking at tau 0.3 (B2 on the full split) robust?

On all 1,319 GSM8K test questions, one seed, D5P4 beta 20 beat DBS alpha 80 by +2.6 [+0.5, +4.8] pass@4 at tau 0.3
(and tied at tau 0.1). Two waves, same task profile as B1 / B2 (`gsm8k_tempered`, 4 x 4, T 1.0):

- `b6_seed1` (`seed_runs`): seed 1 of the tau 0.3 setting of `b2_full` on all four shards: tempered independent
  sampling, DBS alpha 80 and D5P4 beta 20. Shard 0's independent run is B1's `ind_t1_tau0p3_seed1_s0of4`, identical
  in config, so it is not rerun: 11 runs. Shards come in order, so a partial wave covers whole shards.
- `b6_sweep` (`sweep_runs`): DBS alpha 20 / 320 and D5P4 beta 5 / 80 at tau 0.3, shard 0, seed 0, so that the
  comparison on shard 0 is front against front (alpha 80 / beta 20 are in `b2`, the independent run in `b1`).
"""

from __future__ import annotations

from d5p4.waves.b1 import GSM8K_SHARDS, GSM8K_TEMPERED, tempered, tempered_remasking
from d5p4.waves.b2 import search_runs
from d5p4.waves.spec import Run, check_unique_tags, d5p4, diverse_beam, seeded, sharded


TAU = 0.3
SECOND_SEED = 1
SWEEP_SEED = 0
SWEEP_DIVERSITY_ALPHAS = (20.0, 320.0)
SWEEP_W_INTERACTIONS = (5.0, 80.0)


def seed_runs() -> list[Run]:
    result = []
    for shard in range(GSM8K_SHARDS):
        independent_runs = [] if shard == 0 else [tempered(GSM8K_TEMPERED, TAU)]  # shard 0 is in B1
        result.extend(
            sharded(seeded(run, SECOND_SEED), shard, GSM8K_SHARDS) for run in (*independent_runs, *search_runs(TAU))
        )
    return check_unique_tags(result)


def sweep_runs() -> list[Run]:
    search = [
        *(diverse_beam(GSM8K_TEMPERED, alpha) for alpha in SWEEP_DIVERSITY_ALPHAS),
        *(d5p4(GSM8K_TEMPERED, w) for w in SWEEP_W_INTERACTIONS),
    ]
    return check_unique_tags(
        [sharded(seeded(tempered_remasking(run, TAU), SWEEP_SEED), 0, GSM8K_SHARDS) for run in search],
    )

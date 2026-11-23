"""B2: does search still add on top of tempered remasking? GSM8K only.

In B1, independent sampling with tempered remasking (unmasking order sampled from every masked position's confidence
at temperature tau, Olausson et al. 2026) gained about 8 points of pass@4 on GSM8K at much lower Self-BLEU than any
search point under confidence remasking. B2 runs Diverse Beam Search (alpha 80) and D5P4 (beta 20), the pre-specified
points, with the same tempered remasking at tau 0.1 and 0.3. Same task profile, questions (shard 0 of 4) and seed (0)
as B1's tempered independent runs, which are the paired baselines: `b1/results/gsm8k_tempered/ind_t1_tau*_seed0_s0of4`.
This wave has no independent run of its own, so its fronts summary skips the task; compare against B1 per question.
"""

from __future__ import annotations

from d5p4.waves.b1 import GSM8K_SHARDS, GSM8K_TEMPERED, tempered_remasking
from d5p4.waves.spec import Run, check_unique_tags, d5p4, diverse_beam, seeded, sharded


TAUS = (0.1, 0.3)
DIVERSITY_ALPHA = 80.0
W_INTERACTION = 20.0
SEED = 0


def runs() -> list[Run]:
    search = (diverse_beam(GSM8K_TEMPERED, DIVERSITY_ALPHA), d5p4(GSM8K_TEMPERED, W_INTERACTION))
    return check_unique_tags(
        [sharded(seeded(tempered_remasking(run, tau), SEED), 0, GSM8K_SHARDS) for tau in TAUS for run in search],
    )

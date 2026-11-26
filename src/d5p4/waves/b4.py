"""B4: open-ended MDLM generation, every method as a front over its own knob, paired by prompt (reviewer JiJC).

The paper's MDLM figures came from Optuna sweeps of unconditional generation (one study per method, 8 batches per
trial), MAUVE was computed for D5P4 only, against FineWeb-Edu validation text although the paper says FineWeb, and the
search methods' self-cosine counted every lineage group_size times (the last step copies each survivor). B4 replaces
them with fixed grids, the same prompts for every method, three seeds and all metrics for every point.

Setting: mdlm-owt, 1024 positions, 1024 denoising steps (one token per step in expectation, as in the paper), 8 groups
of 4 (pool 32, k = 8 texts per prompt). Prompts are held-out OpenWebText chunks (the last archive of the corpus,
inside MDLM's 100k-document validation split): BOS and the first 32 tokens stay fixed and MDLM writes the other 991.
The MAUVE reference is 5,000 other held-out chunks, cut the same way (see `d5p4.data.owt_heldout`). Perplexity is
GPT-2 Large, MAUVE uses GPT-2 Large features (Pillutla et al. defaults), self-cosine uses the Jina embedder of the
paper, Self-BLEU is within each prompt's k texts (`d5p4.waves.open_eval`, `d5p4.waves.open_fronts`).

Grid, fixed before any run on these prompts, from the ranges the paper's sweeps covered: independent sampling at five
temperatures (the old "CAT" front was quality-only search with a temperature, not independent sampling), Greedy Beam
Search (quality only: the alpha = beta = 0 end of both search fronts), Diverse Beam Search over alpha and D5P4 (additive
kernel) over beta in half-decade steps (beta 3 to 1000 is the paper's MAUVE range). The multiplicative D5P4x comes
last, so the core fronts do not wait for it.

Runs are ordered by prompt shard (strided halves of 128 prompts), then seed, then point: indices 0-53 are 64 prompts
x 3 seeds for the 18 core points, enough for every front and test; 54-107 add the other 64 prompts; 108-131 are
D5P4x on both halves. Each run is one prompt shard x one seed of one point.
"""

from __future__ import annotations

from d5p4.waves.spec import (
    Run,
    Task,
    check_unique_tags,
    d5p4,
    d5p4_multiplicative,
    diverse_beam,
    greedy_beam,
    independent,
    seeded,
    sharded,
)


STEPS = 1024
PROMPTS = 128
PROMPT_SHARDS = 2
SEEDS = (0, 1, 2)

MDLM_OWT = Task(
    name="mdlm_owt",
    entry="mdlm_open.py",
    family="open",
    limit_key="open_num_prompts",
    n_groups=8,
    group_size=4,
    overrides={
        "model": "mdlm",
        "mdlm_steps": STEPS,
        "subsample_start": 0,
        "subsample_end": STEPS,
        "transversal": True,
        "_score_method": "entropy",
        "open_prompt_tokens": 32,
        "open_num_prompts": PROMPTS,
        "open_num_references": 5000,
        "open_heldout_members": 32,
        "skip_eval": True,
    },
)

IND_TEMPERATURES = (0.9, 1.0, 1.05, 1.1, 1.2)
DIVERSITY_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
W_INTERACTIONS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
W_MULTIPLICATIVE = (0.01, 0.1, 1.0, 10.0)


def core_points(task: Task = MDLM_OWT) -> list[Run]:
    """The 18 core points: 5 independent temperatures, Greedy Beam Search, 6 DBS alphas, 6 D5P4 betas."""
    return [
        *(independent(task, t) for t in IND_TEMPERATURES),
        greedy_beam(task),
        *(diverse_beam(task, a) for a in DIVERSITY_ALPHAS),
        *(d5p4(task, w) for w in W_INTERACTIONS),
    ]


def multiplicative_points(task: Task = MDLM_OWT) -> list[Run]:
    return [d5p4_multiplicative(task, w) for w in W_MULTIPLICATIVE]


def _replicates(points: list[Run]) -> list[Run]:
    return [
        sharded(seeded(point, seed), shard, PROMPT_SHARDS)
        for shard in range(PROMPT_SHARDS)
        for seed in SEEDS
        for point in points
    ]


def runs() -> list[Run]:
    return check_unique_tags([*_replicates(core_points()), *_replicates(multiplicative_points())])

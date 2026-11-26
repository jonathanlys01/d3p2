"""A2: mechanism ablations of D5P4 on LLaDA, at the pre-specified point beta 20, T 1.0, confidence remasking.

Each run changes one part of D5P4 (`d5p4(task, 20)`) and keeps everything else:

- `d5p3_w20_t1`: no partition (`transversal=false`): the k survivors are picked from the whole pool, so two may
  share a parent. D5P3 was only evaluated on MDLM so far.
- `d5p4_w20_t1_q{uniform,exp,zscore}`: the quality term of the kernel (`_score_normalization`, see
  `d5p4.subsample.base.normalize_scores`). "uniform" removes the quality signal (Q = 0.5); "exp" uses
  exp(mean negative entropy), with no pool-wise step, so neither the zero-quality candidate nor the step-dependent
  scale of the pool-wise min-max exists; "zscore" keeps a pool-wise step but scales by the spread, not the extremes.
- Checks, at the end so they can be left out: `d5p4_w0_t1` (beta 0) and `rnd_t1` (one random child per parent).
  With beta 0 the kernel is diag(Q) and greedy MAP under the partition is the per-group argmax of Q, which is
  exactly transversal Greedy Beam Search (`gbs_t1`; see `tests/test_waves.py`). A random child per parent is one
  step of the plain sampler, so `rnd_t1` is k independent chains, the same distribution as independent sampling
  scored as random k of N. Both only confirm the code paths.

Questions: GSM8K test shard 0 of 4 (330, 4 x 4) and TruthfulQA shard 0 of 4 (205, 3 x 3), seed 0. Same task
profiles as the paired references, which are not rerun: P1 `p1/results/gsm8k/{d5p4_w20,dbs_a80,gbs,ind}_t1_s0of4`,
and the transfer wave's full-split TruthfulQA runs `transfer/results/truthful_qa/{d5p4_w20,dbs_a80,gbs,ind}_t1`,
restricted to shard 0 (question index % 4 == 0), as the scale pilot did. The wave has no independent run, so its
fronts summary skips both tasks; compare per question against the references.
"""

from __future__ import annotations

from d5p4.waves import p1, transfer
from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, sharded, value_tag


W_INTERACTION = 20.0
SHARDS = 4
QUALITY_NORMALIZATIONS = ("uniform", "exp", "zscore")


def no_partition(task: Task, w_interaction: float = W_INTERACTION) -> Run:
    """D5P3: D5P4 without the partition constraint."""
    run = d5p4(task, w_interaction)
    return Run(
        task=task,
        tag=run.tag.replace("d5p4_", "d5p3_", 1),
        method=run.method,
        overrides={**run.overrides, "transversal": False},
    )


def quality_ablation(task: Task, normalization: str, w_interaction: float = W_INTERACTION) -> Run:
    run = d5p4(task, w_interaction)
    return Run(
        task=task,
        tag=f"{run.tag}_q{normalization}",
        method=run.method,
        overrides={**run.overrides, "_score_normalization": normalization},
    )


def random_transversal(task: Task, temperature: float = 1.0) -> Run:
    """One uniformly random child per parent at every step."""
    return Run(
        task=task,
        tag=f"rnd_t{value_tag(temperature)}",
        method="random",
        overrides={
            "method": "random",
            "n_groups": task.n_groups,
            "group_size": task.group_size,
            "cat_temperature": temperature,
        },
    )


def ablation_runs(task: Task) -> list[Run]:
    return [no_partition(task), *(quality_ablation(task, name) for name in QUALITY_NORMALIZATIONS)]


def check_runs(task: Task) -> list[Run]:
    return [d5p4(task, 0.0), random_transversal(task)]


def runs() -> list[Run]:
    gsm8k, truthful_qa = p1.GSM8K, transfer.TRUTHFUL_QA
    ordered = [
        *ablation_runs(gsm8k),
        *ablation_runs(truthful_qa),
        *check_runs(truthful_qa),
        *check_runs(gsm8k),
    ]
    return check_unique_tags([sharded(run, 0, SHARDS) for run in ordered])

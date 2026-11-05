"""B1: other remasking rules on GSM8K, and tempered remasking as a baseline.

- `gsm8k_stochastic`: the frozen 12-run front under stochastic remasking (positions to unmask sampled among the 2k
  most confident at temperature 0.1), the second remasking rule of the earlier GSM8K tables.
- `gsm8k_tempered`, `truthful_qa_tempered`: independent sampling where the unmasking order is sampled from the
  confidence of every masked position at temperature tau (tempered remasking, "A Tale of Two Temperatures",
  Olausson et al. 2026), next to independent sampling with confidence remasking. Remasking temperature is then a
  diversity knob of its own, to compare with the search fronts.

Same questions as P1 and the transfer wave. Two seeds; seed 0 of every run comes first.
"""

from __future__ import annotations

from d5p4.waves import p1, transfer
from d5p4.waves.spec import Run, Task, check_unique_tags, front_runs, independent, seeded, sharded, value_tag, variant


SEEDS = (0, 1)
GSM8K_SHARDS = 4
TEMPERED_TAUS = (0.03, 0.1, 0.3)

GSM8K_STOCHASTIC = variant(p1.GSM8K, "gsm8k_stochastic", remasking="selection_temperature", selection_temperature=0.1)
GSM8K_TEMPERED = variant(p1.GSM8K, "gsm8k_tempered")
TRUTHFUL_QA_TEMPERED = variant(transfer.TRUTHFUL_QA, "truthful_qa_tempered")


def tempered(task: Task, tau: float) -> Run:
    run = independent(task, 1.0)
    return Run(
        task=task,
        tag=f"{run.tag}_tau{value_tag(tau)}",
        method=run.method,
        overrides={
            **run.overrides,
            "remasking": "selection_temperature",
            "selection_temperature": tau,
            "selection_candidates": 0,
        },
    )


def tempered_runs(task: Task) -> list[Run]:
    return [independent(task, 1.0), *(tempered(task, tau) for tau in TEMPERED_TAUS)]


def runs() -> list[Run]:
    result: list[Run] = []
    for seed in SEEDS:
        result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in front_runs(GSM8K_STOCHASTIC))
        result.extend(sharded(seeded(run, seed), 0, GSM8K_SHARDS) for run in tempered_runs(GSM8K_TEMPERED))
        result.extend(seeded(run, seed) for run in tempered_runs(TRUTHFUL_QA_TEMPERED))
    return check_unique_tags(result)

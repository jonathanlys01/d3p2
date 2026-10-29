"""Declarative experiment waves: a wave is an ordered list of runs, each one generation job.

A run's position in its wave is its Slurm array index, so a wave's run order must stay stable once submitted.
Each run writes to `<root>/results/<task>/<tag>/` and resumes from `<root>/resume/<task>/<tag>/`; the evaluation
scripts in `.scripts_next/` then read `<root>/results/<task>/` and write `<root>/eval/<task>/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DONE_MARKER = ".wave_done"
EVAL_FAMILIES = {"math", "qa", "code"}


@dataclass(frozen=True)
class Task:
    """A benchmark profile shared by every run of a task."""

    name: str
    entry: str  # generation script under src/d5p4
    family: str  # selects the evaluation script, one of EVAL_FAMILIES
    limit_key: str  # override that caps the number of questions, for smoke runs
    n_groups: int
    group_size: int
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.family in EVAL_FAMILIES, f"Unknown evaluation family: {self.family}"

    @property
    def k(self) -> int:
        """Final proposals per question: one per group, or the top k of the independent pool."""
        return self.n_groups

    @property
    def pool_size(self) -> int:
        return self.n_groups * self.group_size


@dataclass(frozen=True)
class Run:
    task: Task
    tag: str  # unique within the task; names the result directory
    method: str
    overrides: dict[str, Any]

    def all_overrides(self) -> dict[str, Any]:
        return {**self.task.overrides, **self.overrides}

    def results_dir(self, root: Path) -> Path:
        return root / "results" / self.task.name / self.tag

    def resume_dir(self, root: Path) -> Path:
        return root / "resume" / self.task.name / self.tag

    def is_done(self, root: Path) -> bool:
        return (self.results_dir(root) / DONE_MARKER).exists()


def value_tag(value: float) -> str:
    """Format a parameter value for a directory name: 0.7 -> '0p7', 20.0 -> '20'."""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def independent(task: Task, temperature: float = 1.0) -> Run:
    """Independent sampling with the same candidate pool as the grouped methods."""
    return Run(
        task=task,
        tag=f"ind_t{value_tag(temperature)}",
        method="baseline",
        overrides={
            "method": "baseline",
            "n_groups": task.pool_size,
            "group_size": 1,
            "cat_temperature": temperature,
        },
    )


def d5p4(task: Task, w_interaction: float, temperature: float = 1.0) -> Run:
    return Run(
        task=task,
        tag=f"d5p4_w{value_tag(w_interaction)}_t{value_tag(temperature)}",
        method="greedy_map",
        overrides={
            "method": "greedy_map",
            "n_groups": task.n_groups,
            "group_size": task.group_size,
            "cat_temperature": temperature,
            "_kernel_type": "cosine",
            "_kernel_method": "additive",
            "_w_interaction": w_interaction,
        },
    )


def diverse_beam(task: Task, alpha: float, temperature: float = 1.0) -> Run:
    return Run(
        task=task,
        tag=f"dbs_a{value_tag(alpha)}_t{value_tag(temperature)}",
        method="diverse_beam",
        overrides={
            "method": "diverse_beam",
            "n_groups": task.n_groups,
            "group_size": task.group_size,
            "cat_temperature": temperature,
            "_diversity_alpha": alpha,
        },
    )


def greedy_beam(task: Task, temperature: float = 1.0) -> Run:
    return Run(
        task=task,
        tag=f"gbs_t{value_tag(temperature)}",
        method="greedy_beam",
        overrides={
            "method": "greedy_beam",
            "n_groups": task.n_groups,
            "group_size": task.group_size,
            "cat_temperature": temperature,
        },
    )


def sharded(run: Run, index: int, count: int) -> Run:
    """Question shard `index` of `count` of a run (strided; see d5p4.data.splits.shard_rows).

    Shards of one run share its tag up to the `_s{index}of{count}` suffix, so a draft on one shard can be extended
    with the others later without regenerating it.
    """
    return Run(
        task=run.task,
        tag=f"{run.tag}_s{index}of{count}",
        method=run.method,
        overrides={**run.overrides, "qa_num_shards": count, "qa_shard_index": index},
    )


# Quality-diversity fronts: grid frozen on 28 October 2026 after the M0 grid-edge runs. Diverse Beam Search gains no
# diversity past alpha 160; D5P4's knee is at beta 20 and it keeps gaining diversity up to 320 at flat quality.
FRONT_TEMPERATURES = (0.7, 1.0, 1.3)
FRONT_DIVERSITY_ALPHAS = (5.0, 20.0, 80.0, 320.0)
FRONT_W_INTERACTIONS = (5.0, 20.0, 80.0, 320.0)


def front_runs(task: Task) -> list[Run]:
    """The 12 runs of one front: independent sampling at 3 temperatures, Greedy Beam Search, 4 DBS and 4 D5P4."""
    return [
        *(independent(task, temperature) for temperature in FRONT_TEMPERATURES),
        greedy_beam(task),
        *(diverse_beam(task, alpha) for alpha in FRONT_DIVERSITY_ALPHAS),
        *(d5p4(task, w) for w in FRONT_W_INTERACTIONS),
    ]


def search_runs(task: Task, temperature: float) -> list[Run]:
    """The search half of a front at another temperature: 4 DBS and 4 D5P4 runs."""
    return [
        *(diverse_beam(task, alpha, temperature) for alpha in FRONT_DIVERSITY_ALPHAS),
        *(d5p4(task, w, temperature) for w in FRONT_W_INTERACTIONS),
    ]


def generation_length(tokens: int) -> dict[str, int]:
    """LLaDA pure diffusion: one block, one token unmasked per step, the DPP window over every step."""
    return {"gen_length": tokens, "llada_steps": tokens, "block_length": tokens, "subsample_end": tokens}


def check_unique_tags(runs: list[Run]) -> list[Run]:
    seen: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.task.name, run.tag)
        assert key not in seen, f"Duplicate run {key}"
        seen.add(key)
    return runs

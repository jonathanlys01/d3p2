"""A3: dump the selection inputs of short LLaDA runs, for exact-vs-greedy MAP and quality-proxy analyses.

Reviewer JiJC asks for exact-versus-greedy gaps at the operating shapes (27, 256 and 65,536 transversals at 3 x 3,
4 x 4 and 8 x 4) and for quality-proxy correlations on partially masked states. Each run sets `selection_dump=true`,
which writes every step's kernel, raw quality scores and selected rows to `<results>/selection_dumps/` (see
`d5p4.selection_dump`), then `python -m d5p4.selection_dump_analysis solver|proxy` reads them offline.

50 questions per run, seed 0, D5P4 at the pre-specified beta 20:

- GSM8K 8 x 4 and 4 x 4: the first 50 questions of P1's shard 0 (`qa_dataset_len=200`, shard 0 of 4), so the 4 x 4
  run repeats the first 50 questions of `p1/results/gsm8k/d5p4_w20_t1_s0of4` with the same seed and serves as a check
  that dumping does not change generation;
- TruthfulQA 3 x 3: the first 50 questions of the transfer wave's run.

Independent sampling with the same pools (GSM8K 16, TruthfulQA 9) is added for the quality proxy: its lineages are
not selected on Q, so its step-t correlations are free of the selection bias that D5P4's surviving lineages carry. It
dumps scores and ancestry only. The heaviest run comes first.
"""

from __future__ import annotations

from dataclasses import replace

from d5p4.waves import p1, transfer
from d5p4.waves.spec import Run, Task, check_unique_tags, d5p4, independent, sharded, variant


QUESTIONS = 50
GSM8K_SHARDS = 4
W_INTERACTION = 20.0
DUMP = {"selection_dump": True, "selection_dump_every": 1}

GSM8K = variant(p1.GSM8K, "gsm8k", qa_dataset_len=QUESTIONS * GSM8K_SHARDS, **DUMP)
GSM8K_K8W4 = replace(GSM8K, name="gsm8k_k8w4", n_groups=8, group_size=4)
TRUTHFUL_QA = variant(transfer.TRUTHFUL_QA, "truthful_qa", qa_dataset_len=QUESTIONS, **DUMP)


def _gsm8k(run: Run) -> Run:
    return sharded(run, 0, GSM8K_SHARDS)


def task_runs(task: Task) -> list[Run]:
    return [d5p4(task, W_INTERACTION), independent(task, 1.0)]


def runs() -> list[Run]:
    return check_unique_tags(
        [
            _gsm8k(d5p4(GSM8K_K8W4, W_INTERACTION)),
            *(_gsm8k(run) for run in task_runs(GSM8K)),
            *task_runs(TRUTHFUL_QA),
        ],
    )

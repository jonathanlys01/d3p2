import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from d5p4.config import Config
from d5p4.math_resume_snapshot import (
    SnapshotThresholdNotMet,
    build_snapshot_payload,
    discover_resume_runs,
    export_run_snapshot,
    inspect_resume_db,
    read_snapshot_rows,
)
from d5p4.result_schema import GenerationResult
from d5p4.resume_db import ResumableRunStore, make_work_items


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> Config:
    return Config(
        disable_sys_args=True,
        model="llada",
        qa_dataset="gsm8k",
        qa_dataset_len=-1,
        qa_n_shots=4,
        seed=42,
        llada_decoder="diffusion",
        force_left_to_right=True,
        method="baseline",
        n_groups=3,
        group_size=1,
        cfg_scale=1.0,
        gen_length=4,
        llada_steps=4,
        block_length=4,
        results_dir=str(tmp_path / "generation-results"),
        resume_db_dir=str(tmp_path / "resume"),
        resume_runs=True,
        resume_db_keep_completed=True,
        standalone_job=True,
    )


def _work_items() -> list[dict]:
    return make_work_items(
        3,
        prefix="gsm8k",
        prompts=["What is 1?", "What is 2?", "What is 3?"],
        references=[["1"], ["2"], ["3"]],
        metadata=[
            {"gold_answer": "1", "answer_str": "one", "item_key": "gsm8k:0"},
            {"gold_answer": "2", "answer_str": "two", "item_key": "gsm8k:1"},
            {"gold_answer": "3", "answer_str": "three", "item_key": "gsm8k:2"},
        ],
    )


def _record_decoded_row(store: ResumableRunStore, item_index: int, gold: str) -> None:
    decoded = [f"The answer is {gold}", "wrong", f"Therefore {gold}"]
    store.record_generated(
        item_index=item_index,
        token_ids=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        prompt_len=1,
        internal_scores=[0.9, 0.1, 0.8],
        generation_metadata={"wall_time_s": 2.0, "model_forward_passes": 4},
    )
    # Leave result=None to prove the snapshot can score decoded rows written by
    # a generation-only run as well as rows pre-scored by llada_math.py.
    store.record_decoded(item_index=item_index, decoded=decoded, result=None)


def test_live_resume_snapshot_is_read_only_and_exports_normal_math_payload(tmp_path):
    config = _config(tmp_path)
    items = _work_items()
    store = ResumableRunStore(
        config=config,
        workflow_id="math_generation:llada",
        mode="math_generation",
        work_items=items,
    )
    store.open()
    try:
        _record_decoded_row(store, 0, "1")
        _record_decoded_row(store, 1, "2")

        run = inspect_resume_db(store.db_path)
        assert run is not None
        assert run.arm == "independent_lr"

        discovered = discover_resume_runs(store.db_dir, arms={"independent_lr"})
        assert discovered == {"independent_lr": run}

        snapshot = read_snapshot_rows(run, threshold=2)
        assert snapshot.ready_count == 2
        assert [row["item_index"] for row in snapshot.rows] == [0, 1]

        payload = build_snapshot_payload(snapshot, num_workers=1)
        GenerationResult.model_validate(payload)
        assert len(payload["results"]) == 2
        assert len(payload["text_samples"]) == 2
        assert payload["snapshot"]["read_only"] is True
        assert payload["snapshot"]["threshold"] == 2
        assert payload["snapshot"]["source_run_status"] == "running"
        assert payload["math_metrics"]["pass@1"] == pytest.approx(2 / 3)
        assert payload["math_metrics"]["pass@3"] == 1.0
        assert payload["ranked_metrics"] == {"ranked_pass@1": 1.0, "ranked_pass@3": 1.0}
        assert payload["generation_stats"]["total_model_forward_passes"] == 8

        output_path = export_run_snapshot(
            run,
            threshold=2,
            results_dir=tmp_path / "snapshots",
            num_workers=1,
        )
        assert output_path.parent.name == "independent_lr"
        with output_path.open() as handle:
            saved = json.load(handle)
        assert saved["snapshot"]["source_experiment_hash"] == store.experiment_hash

        manifests = tmp_path / "manifests"
        manifests.mkdir()
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / ".scripts_next" / "discover_jz_eval_inputs.py"),
                "--root",
                str(tmp_path / "snapshots"),
                "--baseline-dirs",
                str(manifests / "text-baseline.txt"),
                "--math-baseline-dirs",
                str(manifests / "math-baseline.txt"),
                "--subsample-files",
                str(manifests / "text-subsample.txt"),
                "--math-subsample-dirs",
                str(manifests / "math-subsample.txt"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        assert (manifests / "math-baseline.txt").read_text().strip() == str(output_path.parent)

        # The exporter neither marks the source complete nor disturbs its live
        # writer lock/connection.
        status = store.conn.execute(
            "SELECT status FROM runs WHERE experiment_hash = ?",
            (store.experiment_hash,),
        ).fetchone()[0]
        assert status == "running"
        _record_decoded_row(store, 2, "3")
        assert store.generated_indices() == {0, 1, 2}
    finally:
        store.close()


def test_snapshot_requires_the_complete_contiguous_prefix(tmp_path):
    config = _config(tmp_path)
    items = _work_items()
    with ResumableRunStore(
        config=config,
        workflow_id="math_generation:llada",
        mode="math_generation",
        work_items=items,
    ) as store:
        _record_decoded_row(store, 1, "2")
        run = inspect_resume_db(store.db_path)
        assert run is not None

        with pytest.raises(SnapshotThresholdNotMet, match="missing indices: 0"):
            read_snapshot_rows(run, threshold=2)


def test_read_only_inspection_does_not_create_or_migrate_schema(tmp_path):
    db_path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")
        connection.execute("INSERT INTO unrelated VALUES (1)")

    with pytest.raises(sqlite3.OperationalError, match="no such table: runs"):
        inspect_resume_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert tables == {"unrelated"}

import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from d5p4.diffusion_llada import cfg_combine_logits, cfg_is_active
from d5p4.math_shard_merge import MathShardMergeError, merge_math_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / ".jz_next" / "gsm8k_4x4_paper_sharded.slurm"


def _config(shard_index: int) -> dict:
    return {
        "model": "llada",
        "llada_model_path": "/work/models/GSAI-ML/LLaDA-8B-Instruct",
        "qa_dataset": "gsm8k",
        "qa_dataset_len": 4,
        "qa_n_shots": 0,
        "qa_num_shards": 4,
        "qa_shard_index": shard_index,
        "method": "baseline",
        "n_groups": 16,
        "group_size": 1,
        "cat_temperature": 1.0,
        "remasking": "low_confidence",
        "cfg_scale": 0.0,
        "llada_reference_cfg": True,
        "llada_steps": 256,
        "gen_length": 256,
        "block_length": 256,
        "standalone_job": True,
        "skip_eval": False,
        "results_dir": f"/tmp/results/shard_{shard_index}",
        "resume_db_dir": f"/tmp/resume/shard_{shard_index}",
        "comment": f"shard {shard_index}",
    }


def _payload(shard_index: int) -> dict:
    gold = str(shard_index + 1)
    generations = [f"The answer is {gold}."] + ["The answer is 999."] * 15
    correctness = [1] + [0] * 15
    internal_scores = [float(index) for index in range(16)]
    result = {
        "dataset_index": shard_index,
        "question": f"Question: What is {gold}?\nAnswer:",
        "gold_answer": gold,
        "answer_str": gold,
        "generations": generations,
        "scores": correctness,
        "accuracy": 1 / 16,
    }
    return {
        "config": _config(shard_index),
        "experiment_id": f"experiment-{shard_index}",
        "dataset_indices": [shard_index],
        "text_samples": [generations],
        "references": [[gold]],
        "internal_scores": [internal_scores],
        "internal_score_metadata": {"higher_is_better": True},
        "results": [result],
        "generation_metadata": [{"wall_time_s": 1.0, "model_forward_passes": 256}],
    }


def _dry_run(task_id: int) -> str:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "WORK_ejh": "/work/ejh",
        "SCRATCH_ejh": "/scratch/ejh",
        "SLURM_ARRAY_TASK_ID": str(task_id),
        "SLURM_ARRAY_JOB_ID": "1234",
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_reference_cfg_zero_is_conditional_without_guidance():
    conditional = torch.tensor([[2.0, 4.0]])
    unconditional = torch.tensor([[1.0, 1.5]])

    assert cfg_is_active(0.0, reference_convention=True) is False
    assert cfg_is_active(0.5, reference_convention=True) is True
    assert cfg_is_active(0.0, reference_convention=False) is True
    assert torch.equal(
        cfg_combine_logits(conditional, unconditional, 0.0, reference_convention=True),
        conditional,
    )
    assert torch.equal(
        cfg_combine_logits(conditional, unconditional, 2.0, reference_convention=True),
        unconditional + 3.0 * (conditional - unconditional),
    )


def test_launcher_maps_four_methods_to_four_independent_shards():
    expected = {
        0: ("baseline", 0),
        3: ("baseline", 3),
        4: ("greedy_map", 0),
        6: ("greedy_map", 2),
        8: ("diverse_beam", 0),
        12: ("greedy_beam", 0),
        15: ("greedy_beam", 3),
    }
    for task_id, (method, shard_index) in expected.items():
        output = _dry_run(task_id)
        assert f"method={method}" in output
        assert f"qa_shard_index={shard_index}" in output
        assert f"shards/shard_{shard_index}" in output
        assert f"resume/shard_{shard_index}" in output
        assert "--expected-candidates=16" in output
        assert f"--expected-method={method}" in output


def test_launcher_locks_paper_like_zero_shot_low_confidence_profile():
    output = _dry_run(4)
    launcher_text = LAUNCHER.read_text()

    assert "#SBATCH --account=ejh@h100" in launcher_text
    assert "#SBATCH --array=0-15%16" in launcher_text
    assert "#SBATCH --gres=gpu:1" in launcher_text
    assert 'export WORK="${WORK_ejh:' in launcher_text
    assert 'export SCRATCH="${SCRATCH_ejh:' in launcher_text
    assert "qa_n_shots=0" in output
    assert "LLaDA-8B-Instruct" in output
    assert "remasking=low_confidence" in output
    assert "cfg_scale=0" in output
    assert "llada_reference_cfg=true" in output
    assert "cat_temperature=1.0" in output
    assert "llada_steps=256" in output
    assert "gen_length=256" in output
    assert "block_length=256" in output
    assert "selection_temperature" not in launcher_text or "selection_temperature=0.0" in output
    assert "remasking=random" not in launcher_text
    assert "remasking=selection_temperature" not in launcher_text
    assert 'if "${MERGE_COMMAND[@]}" 2>/dev/null; then' in launcher_text


def test_generic_merge_orders_strided_rows_and_keeps_sixteen_candidates():
    payload = merge_math_shards(
        [_payload(index) for index in range(4)],
        world_size=4,
        expected_method="baseline",
        expected_candidates=16,
        num_workers=1,
    )

    assert payload["dataset_indices"] == [0, 1, 2, 3]
    assert payload["shard_merge"]["world_size"] == 4
    assert len(payload["text_samples"]) == 4
    assert all(len(group) == 16 for group in payload["text_samples"])
    assert payload["config"]["cfg_scale"] == 0.0
    assert payload["config"]["llada_reference_cfg"] is True
    assert "qa_num_shards" not in payload["config"]
    assert "qa_shard_index" not in payload["config"]
    assert payload["math_metrics"]["pass@16"] == 1.0
    assert payload["comparison_metrics"]["pass@1"] == pytest.approx(1 / 16)


def test_generic_merge_accepts_mixed_single_and_distributed_runtime_topology():
    payloads = [_payload(index) for index in range(4)]
    payloads[2]["config"]["standalone_job"] = False

    payload = merge_math_shards(
        payloads,
        world_size=4,
        expected_method="baseline",
        expected_candidates=16,
        num_workers=1,
    )

    assert payload["dataset_indices"] == [0, 1, 2, 3]


def test_generic_merge_rejects_wrong_method_and_missing_index():
    payloads = [_payload(index) for index in range(4)]
    with pytest.raises(MathShardMergeError, match="expected 'greedy_map'"):
        merge_math_shards(payloads, world_size=4, expected_method="greedy_map", num_workers=1)

    missing = deepcopy(payloads)
    missing[3]["dataset_indices"] = [7]
    missing[3]["results"][0]["dataset_index"] = 7
    with pytest.raises(MathShardMergeError, match="complete index range"):
        merge_math_shards(missing, world_size=4, expected_method="baseline", num_workers=1)

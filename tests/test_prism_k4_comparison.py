import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from d5p4.config import Config
from d5p4.llada_math import _shard_indexed_rows
from d5p4.prism_k4_math_merge import ShardMergeError, merge_payloads
from d5p4.resume_db import semantic_config_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / ".scripts" / "gsm8k_prism_k4.sbatch"


def _locked_config(shard_index: int) -> dict:
    return {
        "model": "llada",
        "qa_dataset": "gsm8k",
        "qa_dataset_len": 4,
        "qa_n_shots": 0,
        "qa_num_shards": 4,
        "qa_shard_index": shard_index,
        "method": "greedy_map",
        "n_groups": 2,
        "group_size": 2,
        "transversal": True,
        "_kernel_method": "additive",
        "_kernel_type": "cosine",
        "_w_interaction": 25.0,
        "subsample_start": 0,
        "subsample_end": 1024,
        "cat_temperature": 0.7,
        "remasking": "low_confidence",
        "selection_temperature": 0.0,
        "cfg_scale": 1.0,
        "llada_steps": 256,
        "gen_length": 256,
        "block_length": 32,
        "standalone_job": True,
        "skip_eval": False,
        "results_dir": f"/tmp/results/shard_{shard_index}",
        "resume_db_dir": f"/tmp/resume/shard_{shard_index}",
        "comment": f"shard {shard_index}",
    }


def _payload(shard_index: int, correctness: list[int], internal_scores: list[float]) -> dict:
    dataset_index = shard_index
    gold = str(dataset_index + 1)
    generations = [
        f"The answer is {gold}." if correct else "The answer is 999."
        for correct in correctness
    ]
    result = {
        "dataset_index": dataset_index,
        "question": f"Question: What is {gold}?\nAnswer:",
        "gold_answer": gold,
        "answer_str": gold,
        "generations": generations,
        "scores": correctness,
        "accuracy": sum(correctness) / len(correctness),
    }
    return {
        "config": _locked_config(shard_index),
        "experiment_id": f"experiment-{shard_index}",
        "text_samples": [generations],
        "references": [[gold]],
        "internal_scores": [internal_scores],
        "internal_score_metadata": {
            "name": "confidence",
            "method": "final_step_mean_token_logprob",
            "scope": "generated_tokens",
            "higher_is_better": True,
        },
        "results": [result],
        "generation_metadata": [{"wall_time_s": 1.0, "model_forward_passes": 256}],
    }


def _four_payloads() -> list[dict]:
    return [
        _payload(0, [1, 0, 0, 0], [0.9, 0.3, 0.2, 0.1]),
        _payload(1, [1, 1, 0, 0], [0.7, 0.6, 0.9, 0.8]),
        _payload(2, [0, 0, 0, 0], [0.4, 0.3, 0.2, 0.1]),
        _payload(3, [1, 1, 1, 1], [0.4, 0.3, 0.2, 0.1]),
    ]


def test_strided_question_shards_are_disjoint_complete_and_balanced():
    rows = list(range(11))
    shards = [_shard_indexed_rows(rows, shard_index=index, num_shards=4) for index in range(4)]

    assert [[row for _, row in shard] for shard in shards] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7],
    ]
    indexed = [item for shard in shards for item in shard]
    assert sorted(index for index, _ in indexed) == list(range(11))
    assert len({index for index, _ in indexed}) == 11


def test_config_validates_question_shard_bounds():
    config = Config(disable_sys_args=True, qa_num_shards=4, qa_shard_index=3)
    assert config.qa_num_shards == 4
    assert config.qa_shard_index == 3

    with pytest.raises(AssertionError, match="qa_shard_index"):
        Config(disable_sys_args=True, qa_num_shards=4, qa_shard_index=4)

    semantic = semantic_config_dict(config)
    assert "qa_num_shards" not in semantic
    assert "qa_shard_index" not in semantic


def test_merge_reports_internal_accuracy_pass_metrics_selection_and_nfe():
    payload = merge_payloads(_four_payloads(), world_size=4, num_workers=1)

    assert payload["shard_merge"]["dataset_indices"] == [0, 1, 2, 3]
    assert payload["comparison_metrics"] == pytest.approx(
        {
            "internal_accuracy": 0.5,
            "pass@1": 0.4375,
            "pass@2": 7 / 12,
        },
    )
    assert payload["ranked_metrics"]["ranked_pass@1"] == 0.5
    assert payload["math_metrics"]["pass@4"] == 0.75
    assert payload["selected_results"][0]["selected_index"] == 0
    assert payload["selected_results"][0]["correct"] is True
    assert payload["selected_results"][1]["selected_index"] == 2
    assert payload["selected_results"][1]["correct"] is False
    assert payload["results"][1]["internal_selection"] == payload["selected_results"][1]
    assert payload["nfe_accounting"]["trajectory_nfe_per_prompt"] == 1024
    assert payload["nfe_accounting"]["per_group_trajectory_nfe_per_prompt"] == 512
    assert payload["nfe_accounting"]["batched_model_forward_calls_per_prompt"] == 256
    assert payload["nfe_accounting"]["observed_mean_model_forward_calls_per_prompt"] == 256


def test_merge_supports_eight_question_shards_and_dynamic_candidate_count():
    payloads = []
    for shard_index in range(8):
        payload = _payload(shard_index, [1] * 8, [0.9 - shard_index * 0.01] * 8)
        payload["config"]["qa_num_shards"] = 8
        payload["config"]["n_groups"] = 2
        payload["config"]["group_size"] = 4
        payloads.append(payload)

    merged = merge_payloads(
        payloads,
        world_size=8,
        n_groups=2,
        group_size=4,
        expected_candidates=8,
        num_workers=1,
    )

    assert len(merged["text_samples"]) == 8
    assert len(merged["text_samples"][0]) == 8
    assert merged["math_metrics"]["pass@8"] == 1.0
    assert merged["config"]["n_groups"] == 2
    assert merged["config"]["group_size"] == 4


def test_merge_rejects_incomplete_or_incompatible_shards():
    missing_index = deepcopy(_four_payloads())
    missing_index[3]["results"][0]["dataset_index"] = 7
    with pytest.raises(ShardMergeError, match="complete index range"):
        merge_payloads(missing_index, world_size=4, num_workers=1)

    wrong_cardinality = deepcopy(_four_payloads())
    wrong_cardinality[0]["text_samples"][0].pop()
    with pytest.raises(ShardMergeError, match="3 candidates"):
        merge_payloads(wrong_cardinality, world_size=4, num_workers=1)

    wrong_config = deepcopy(_four_payloads())
    wrong_config[2]["config"]["cfg_scale"] = 2.5
    with pytest.raises(ShardMergeError, match="cfg_scale"):
        merge_payloads(wrong_config, world_size=4, num_workers=1)


def test_local_array_launcher_dry_run_locks_question_sharded_prism_profile():
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "ROOT": str(REPO_ROOT),
            "USER": "test-user",
            "SLURM_ARRAY_TASK_ID": "2",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_ARRAY_JOB_ID": "1234",
        },
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    output = completed.stdout

    assert "srun --nodes=1 --ntasks=1 --cpus-per-task=8 --kill-on-bad-exit=1" in output
    assert "standalone_job=true" in output
    assert "qa_num_shards=4" in output
    assert "qa_shard_index=2" in output
    assert "results_dir=" in output and "shards/shard_2" in output
    assert "resume_db_dir=" in output and "resume/shard_2" in output
    assert "method=greedy_map" in output
    assert "n_groups=2" in output
    assert "group_size=2" in output
    assert "transversal=true" in output
    assert "cfg_scale=1.0" in output
    assert "cat_temperature=0.7" in output
    assert "remasking=low_confidence" in output
    assert "selection_temperature=0.0" in output
    assert "gen_length=256" in output
    assert "llada_steps=256" in output
    assert "block_length=32" in output
    assert "d5p4.prism_k4_math_merge" in output
    assert "--world-size=4" in output

    launcher_text = LAUNCHER.read_text()
    assert "#SBATCH --array=0-3" in launcher_text
    assert "#SBATCH --ntasks=1" in launcher_text
    assert "#SBATCH --gres=gpu:a100:1" in launcher_text
    assert "%A_%a" in launcher_text
    assert 'qa_shard_index="${TASK_ID}"' in launcher_text
    assert 'if "${MERGE_COMMAND[@]}" 2>/dev/null; then' in launcher_text
    assert "the last" in launcher_text
    assert "threshold 0.95" in launcher_text
    assert "defaults to 0.85" in launcher_text


def test_all_array_elements_have_independent_paths():
    outputs = []
    for shard_index in range(4):
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "ROOT": str(REPO_ROOT),
            "USER": "test-user",
            "SLURM_ARRAY_TASK_ID": str(shard_index),
            "SLURM_ARRAY_TASK_COUNT": "4",
        }
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(completed.stdout)

    for shard_index, output in enumerate(outputs):
        assert f"qa_shard_index={shard_index}" in output
        assert f"shards/shard_{shard_index}" in output
        assert f"resume/shard_{shard_index}" in output

def test_merge_cli_writes_stable_json(tmp_path):
    shard_root = tmp_path / "shards"
    for shard_index, payload in enumerate(_four_payloads()):
        shard_dir = shard_root / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True)
        (shard_dir / f"math-{shard_index}.json").write_text(json.dumps(payload))
    output_path = tmp_path / "math-d5p4-prism-k4.json"

    subprocess.run(
        [
            "python",
            "-m",
            "d5p4.prism_k4_math_merge",
            f"--shard-root={shard_root}",
            f"--output={output_path}",
            "--world-size=4",
            "--num-workers=1",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=True,
    )
    saved = json.loads(output_path.read_text())
    assert saved["comparison_metrics"]["internal_accuracy"] == 0.5
    assert not output_path.with_suffix(".json.tmp").exists()

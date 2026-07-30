import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from d5p4.config import Config
from d5p4.llada_math import (
    _aggregate_generation_metadata,
    _internal_score_metadata,
    _ranked_pass_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".scripts_next" / "gsm8k_classic_beam_comparison.sh"
LOCAL_CLUSTER_SCRIPT = REPO_ROOT / ".scripts" / "gsm8k_block1_sweep.sbatch"
JZ_LTR_SCRIPT = REPO_ROOT / ".jz_next" / "llada_ltr_beam_comparison.slurm"
JZ_D5P4_BEAM_SCRIPT = REPO_ROOT / "_scratchpad" / "llada_d5p4_beam_sharded.slurm"
LOCAL_D5P4_BEAM_SCRIPT = REPO_ROOT / "_scratchpad" / "llada_d5p4_beam_sharded_local.sbatch"


def test_ranked_metrics_use_internal_score_order():
    results = [
        {"scores": [0, 1, 0]},
        {"scores": [1, 0, 0]},
    ]
    internal_scores = [
        [0.1, 0.9, 0.2],
        [0.1, 0.9, 0.2],
    ]

    metrics = _ranked_pass_metrics(results, internal_scores)

    assert metrics == {"ranked_pass@1": 0.5, "ranked_pass@3": 1.0}


def test_generation_metadata_aggregation_and_score_metadata():
    stats = _aggregate_generation_metadata(
        [
            {"wall_time_s": 1.0, "model_forward_passes": 4},
            None,
            {"wall_time_s": 3.0, "model_forward_passes": 6},
        ],
    )

    assert stats == {
        "prompt_count": 3,
        "measured_prompt_count": 2,
        "missing_prompt_count": 1,
        "total_wall_time_s": 4.0,
        "mean_wall_time_s": 2.0,
        "total_model_forward_passes": 10,
        "mean_model_forward_passes": 5.0,
    }

    diffusion = Config(disable_sys_args=True, model="llada")
    classic = Config(
        disable_sys_args=True,
        model="llada",
        llada_decoder="classic_beam",
        cfg_scale=1.0,
        method="ltr_beam",
        transversal=False,
        n_groups=4,
        group_size=1,
    )
    assert _internal_score_metadata(diffusion)["method"] == "final_step_mean_token_logprob"
    assert _internal_score_metadata(classic)["method"] == "length_normalized_left_to_right_token_logprob"


def test_llada_math_classic_beam_reports_pass_and_generation_metrics(monkeypatch):
    from d5p4 import llada_math

    class _Dataset:
        def __len__(self):
            return 1

        def itertuples(self):
            return iter(
                [
                    SimpleNamespace(
                        question="What is 40 + 2?",
                        answer_str="42",
                        answer_number="42",
                    ),
                ],
            )

    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            assert skip_special_tokens
            return "".join(str(token_id) for token_id in token_ids)

    class _Sampler:
        distributed_utils = None
        device = "cpu"
        tokenizer = _Tokenizer()

        def __init__(self, config):
            self.config = config
            self.model = object()
            self.last_forward_count = 0

        def sample(self, prompt, return_internal_scores=False):
            assert prompt == "What is 40 + 2?"
            assert return_internal_scores
            self.last_forward_count = self.config.gen_length
            return torch.tensor([[9, 4, 2], [9, 1, 0]]), torch.tensor([0.9, 0.1])

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            disable_sys_args=True,
            model="llada",
            llada_decoder="classic_beam",
            cfg_scale=1.0,
            method="ltr_beam",
            transversal=False,
            gen_length=2,
            n_groups=2,
            group_size=1,
            qa_dataset="gsm8k",
            qa_dataset_len=1,
            skip_eval=False,
            resume_runs=False,
            results_dir=tmpdir,
            standalone_job=True,
            compile_model=False,
        )
        monkeypatch.setattr(llada_math, "Config", lambda: config)
        monkeypatch.setattr(llada_math, "gsm8k", lambda _config: _Dataset())
        monkeypatch.setattr(llada_math, "LLADASampler", _Sampler)
        monkeypatch.setattr(llada_math, "compile_model", lambda model, _config, dynamic=False: model)
        monkeypatch.setattr(llada_math, "seed_all", lambda _seed: None)
        monkeypatch.setattr(llada_math, "print", lambda *_args, **_kwargs: None)

        llada_math.main()

        output_files = list(Path(tmpdir).glob("math-*.json"))
        assert len(output_files) == 1
        with output_files[0].open() as handle:
            payload = json.load(handle)

    assert payload["math_metrics"]["pass@1"] == 0.5
    assert payload["math_metrics"]["pass@2"] == 1.0
    assert payload["ranked_metrics"] == {"ranked_pass@1": 1.0, "ranked_pass@2": 1.0}
    assert payload["generation_stats"]["total_model_forward_passes"] == 2
    assert payload["generation_stats"]["mean_model_forward_passes"] == 2.0
    assert payload["internal_score_metadata"]["method"] == "length_normalized_left_to_right_token_logprob"


def test_gsm8k_classic_beam_comparison_dry_run_has_three_equal_population_arms():
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "BEAM_SIZE": "12",
            "D5P4_GROUP_SIZE": "3",
            "CLASSIC_BEAM_BRANCHING_FACTOR": "5",
            "LLADA_PYTHON_BIN": "python",
        },
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    commands = [line for line in completed.stdout.splitlines() if "llada_math.py" in line]
    assert len(commands) == 3

    independent, d5p4, classic = commands
    assert "llada_decoder=diffusion" in independent
    assert "method=baseline" in independent
    assert "n_groups=12" in independent
    assert "group_size=1" in independent

    assert "llada_decoder=diffusion" in d5p4
    assert "method=greedy_map" in d5p4
    assert "n_groups=4" in d5p4
    assert "group_size=3" in d5p4
    assert "_kernel_method=additive" in d5p4
    assert "_w_interaction=25.0" in d5p4

    assert "llada_decoder=classic_beam" in classic
    assert "classic_beam_branching_factor=5" in classic
    assert "method=ltr_beam" in classic
    assert "transversal=false" in classic
    assert "n_groups=12" in classic
    assert "group_size=1" in classic

    for command in commands:
        assert "cfg_scale=1.0" in command
        assert "standalone_job=true" in command
        assert "gen_length=256" in command


def test_jz_ltr_comparison_distinguishes_global_and_transversal_beam_methods():
    env = os.environ.copy()
    env.update(
        {
            "WORK_ejh": "/tmp/work",
            "DRY_RUN": "1",
        },
    )
    commands = []
    for task_id in range(4):
        env["SLURM_ARRAY_TASK_ID"] = str(task_id)
        completed = subprocess.run(
            ["bash", str(JZ_LTR_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        commands.extend(line for line in completed.stdout.splitlines() if "llada_math.py" in line)

    assert len(commands) == 4
    independent, classic, d5p4, transversal = commands
    assert "method=baseline" in independent
    assert "method=ltr_beam" in classic
    assert "transversal=false" in classic
    assert "n_groups=9" in classic
    assert "group_size=1" in classic
    assert "method=greedy_map" in d5p4
    assert "method=ltr_beam" in transversal
    assert "transversal=true" in transversal
    assert "n_groups=3" in transversal
    assert "group_size=3" in transversal


def test_d5p4_beam_scratch_launchers_keep_weight_and_layout_namespaces_distinct():
    jz_env = os.environ.copy()
    jz_env.update(
        {
            "WORK_ejh": "/tmp/work",
            "SCRATCH_ejh": "/tmp/scratch",
            "SLURM_JOB_ID": "123",
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "NUM_WORKERS": "4",
            "D5P4_W_INTERACTION": "0",
            "BEAM_LAYOUT": "transversal",
            "DRY_RUN": "1",
        },
    )
    jz = subprocess.run(
        ["bash", str(JZ_D5P4_BEAM_SCRIPT)],
        cwd=REPO_ROOT,
        env=jz_env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "method=greedy_map" in jz
    assert "_w_interaction=0" in jz
    assert "transversal=true" in jz
    assert "n_groups=3 group_size=3" in jz
    assert "transversal_w0" in jz
    assert "--expected-method=greedy_map" in jz
    assert "--expected-transversal=true" in jz
    assert "--expected-weight=0" in jz

    local_env = os.environ.copy()
    local_env.update(
        {
            "ROOT": str(REPO_ROOT),
            "SLURM_JOB_ID": "123",
            "SLURM_GPUS_ON_NODE": "4",
            "NUM_WORKERS": "4",
            "D5P4_W_INTERACTION": "0.5",
            "BEAM_LAYOUT": "global",
            "DRY_RUN": "1",
        },
    )
    local = subprocess.run(
        ["bash", str(LOCAL_D5P4_BEAM_SCRIPT)],
        cwd=REPO_ROOT,
        env=local_env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--ntasks=4" in local
    assert "--gpus-per-task=1" in local
    assert "method=greedy_map" in local
    assert "_w_interaction=0.5" in local
    assert "transversal=false" in local
    assert "n_groups=9 group_size=1" in local
    assert "global_w0p5" in local
    assert "--expected-transversal=false" in local
    assert "--expected-weight=0.5" in local


def test_local_cluster_block1_sweep_uses_one_group_of_three_per_gpu():
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
        },
    )
    commands = []
    for task_id in range(3):
        env["SLURM_ARRAY_TASK_ID"] = str(task_id)
        completed = subprocess.run(
            ["bash", str(LOCAL_CLUSTER_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        commands.extend(line for line in completed.stdout.splitlines() if "llada_math.py" in line)

    assert len(commands) == 3
    independent, d5p4, greedy_beam = commands

    for command in commands:
        assert "srun --exclusive --nodes=1 --ntasks=3 --cpus-per-task=8 --gres=gpu:3" in command
        assert "standalone_job=false" in command
        assert "resume_db_keep_completed=true" in command
        assert "gen_length=256" in command
        assert "qa_n_shots=0" in command
        assert "compile_model=true" in command
        assert "cat_temperature=1.0" in command
        assert "remasking=selection_temperature" in command
        assert "selection_temperature=0.1" in command
        assert "skip_eval=true" in command
        assert "cfg_scale=2.5" in command
        assert "block_length=1" in command
        assert "llada_decoder=" not in command
        assert "force_left_to_right=" not in command

    assert "method=baseline" in independent
    assert "n_groups=3" in independent
    assert "group_size=1" in independent

    assert "method=greedy_map" in d5p4
    assert "n_groups=1" in d5p4
    assert "group_size=3" in d5p4
    assert "subsample_start=0" in d5p4
    assert "subsample_end=1024" in d5p4
    assert "_kernel_method=additive" in d5p4
    assert "_w_interaction=25.0" in d5p4

    assert "method=greedy_beam" in greedy_beam
    assert "n_groups=1" in greedy_beam
    assert "group_size=3" in greedy_beam
    assert "subsample_start=0" in greedy_beam
    assert "subsample_end=1024" in greedy_beam

    script_text = LOCAL_CLUSTER_SCRIPT.read_text()
    assert "#SBATCH --ntasks=3" in script_text
    assert "#SBATCH --gres=gpu:a100:3" in script_text
    assert "#SBATCH --array=0-2" in script_text
    assert "GLOBAL_GROUP_COUNT=3" in script_text
    assert "D5P4_GROUP_SIZE=3" in script_text
    assert "CFG_SCALE=2.5" in script_text
    assert "BLOCK_LENGTH=1" in script_text
    assert "UV_PROJECT_ENVIRONMENT=" in script_text
    assert 'PATH="/usr/bin:/bin:' in script_text

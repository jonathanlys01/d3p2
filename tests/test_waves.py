import json
import subprocess
from pathlib import Path

import pytest

import d5p4.waves.__main__ as cli
from d5p4.waves import m0
from d5p4.waves import get_wave
from d5p4.waves.spec import DONE_MARKER, value_tag


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "src" / "d5p4" / "_default.yaml"


def test_m0_order_is_stable():
    runs = get_wave("m0")

    assert len(runs) == 52
    assert [(run.task.name, run.tag) for run in runs[:4]] == [
        ("gsm8k", "ind_t0p7"),
        ("gsm8k", "ind_t1"),
        ("gsm8k", "ind_t1p3"),
        ("gsm8k", "gbs_t1"),
    ]
    assert (runs[44].task.name, runs[44].tag) == ("mbpp", "d5p4_w160_t1")
    assert [(run.task.name, run.tag) for run in runs[45:]] == [
        ("gsm8k", "dbs_a160_t1"),
        ("gsm8k", "dbs_a320_t1"),
        ("ai2_arc", "dbs_a160_t1"),
        ("ai2_arc", "dbs_a320_t1"),
        ("mbpp", "dbs_a160_t1"),
        ("mbpp", "dbs_a320_t1"),
        ("gsm8k", "d5p4_w320_t1"),
    ]


def test_m0_pools_match_across_methods():
    for run in get_wave("m0"):
        overrides = run.all_overrides()
        assert overrides["n_groups"] * overrides["group_size"] == run.task.pool_size
        assert overrides["dataset_split"] == "dev"
        assert overrides["cfg_scale"] == 0
        assert overrides["llada_reference_cfg"] is True


@pytest.mark.parametrize("run", get_wave("m0"), ids=lambda run: f"{run.task.name}-{run.tag}")
def test_m0_runs_build_valid_configs(run):
    config = cli.build_config(run, DEFAULT_CONFIG)

    assert config.method == run.method
    assert config.dataset_split == "dev"
    assert config.dataset_shuffle_seed == 0


def test_value_tag():
    assert value_tag(0.7) == "0p7"
    assert value_tag(20.0) == "20"
    assert value_tag(160.0) == "160"


def test_run_command_applies_limit_and_output_dirs(tmp_path):
    run = get_wave("m0")[0]
    command = cli.run_command(run, tmp_path, DEFAULT_CONFIG, None, limit=2)

    assert command[1].endswith("src/d5p4/llada_math.py")
    assert f"--config={DEFAULT_CONFIG}" in command
    assert "qa_dataset_len=2" in command
    assert f"results_dir={tmp_path}/results/gsm8k/ind_t0p7" in command
    assert f"resume_db_dir={tmp_path}/resume/gsm8k/ind_t0p7" in command
    assert "cfg_scale=0" in command
    assert "llada_reference_cfg=true" in command


def _fake_subprocess(returncode):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout="abc123\n", stderr="")

    return run, calls


def test_run_marks_success_and_skips_completed_runs(tmp_path, monkeypatch):
    fake_run, calls = _fake_subprocess(0)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    argv = ["run", "--wave", "m0", "--root", str(tmp_path), "--config", str(DEFAULT_CONFIG), "--index", "3"]

    assert cli.main(argv) == 0
    marker = tmp_path / "results" / "gsm8k" / "gbs_t1" / DONE_MARKER
    assert json.loads(marker.read_text())["git_revision"] == "abc123"
    generation_calls = [call for call in calls if call[0] != "git"]
    assert len(generation_calls) == 1

    assert cli.main(argv) == 0
    assert len([call for call in calls if call[0] != "git"]) == 1


def test_failed_run_is_not_marked_done(tmp_path, monkeypatch):
    fake_run, _calls = _fake_subprocess(3)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    argv = ["run", "--wave", "m0", "--root", str(tmp_path), "--config", str(DEFAULT_CONFIG), "--index", "0"]
    assert cli.main(argv) == 3
    assert not (tmp_path / "results" / "gsm8k" / "ind_t0p7" / DONE_MARKER).exists()


def test_status_missing_lists_array_indices(tmp_path, capsys):
    runs = get_wave("m0")
    for run in runs[1:]:
        run.results_dir(tmp_path).mkdir(parents=True)
        (run.results_dir(tmp_path) / DONE_MARKER).write_text("{}")

    cli.main(["status", "--wave", "m0", "--root", str(tmp_path), "--missing"])

    assert capsys.readouterr().out.strip() == "0"


def test_array_spec_compresses_ranges():
    assert cli.array_spec([]) == ""
    assert cli.array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert cli.array_spec([4]) == "4"


def test_summarize_joins_eval_outputs_to_runs(tmp_path):
    math_eval = tmp_path / "eval" / "gsm8k" / "d5p4_w20_t1" / "math-x-math-bon-int.json"
    math_eval.parent.mkdir(parents=True)
    math_eval.write_text(
        json.dumps(
            {
                "source_relative_path": "d5p4_w20_t1",
                "selection_metric": "int",
                "math_metrics": {"pass@4": 0.6, "self_bleu": {"mean": 70.0}},
            },
        ),
    )
    code_eval = tmp_path / "eval" / "mbpp" / "ind_t1" / "code-x-code-bon-random.json"
    code_eval.parent.mkdir(parents=True)
    code_eval.write_text(
        json.dumps({"source_relative_path": "ind_t1/code-x.json", "metrics": {"pass@3": 0.25}}),
    )
    output = tmp_path / "summary.csv"

    assert cli.main(["summarize", "--wave", "m0", "--root", str(tmp_path), "--output", str(output)]) == 0

    rows = output.read_text().splitlines()
    assert len(rows) == 3
    header = rows[0].split(",")
    assert {"math.pass@4", "math.self_bleu.mean", "pass@3", "w_interaction"} <= set(header)
    gsm8k_row = dict(zip(header, rows[1].split(","), strict=True))
    assert gsm8k_row["tag"] == "d5p4_w20_t1"
    assert gsm8k_row["w_interaction"] == "20.0"
    assert gsm8k_row["math.pass@4"] == "0.6"


def test_front_runs_use_the_frozen_grid():
    from d5p4.waves.spec import front_runs

    tags = [run.tag for run in front_runs(m0.GSM8K)]
    assert tags == [
        "ind_t0p7",
        "ind_t1",
        "ind_t1p3",
        "gbs_t1",
        "dbs_a5_t1",
        "dbs_a20_t1",
        "dbs_a80_t1",
        "dbs_a320_t1",
        "d5p4_w5_t1",
        "d5p4_w20_t1",
        "d5p4_w80_t1",
        "d5p4_w320_t1",
    ]


def test_transfer_wave_covers_the_three_new_datasets_on_eval_splits():
    runs = get_wave("transfer")

    assert len(runs) == 44
    assert [(run.task.name, run.tag) for run in runs[36:38]] == [
        ("truthful_qa", "dbs_a5_t1p3"),
        ("truthful_qa", "dbs_a20_t1p3"),
    ]
    assert runs[-1].tag == "d5p4_w320_t1p3"
    assert cli.build_config(runs[-1], DEFAULT_CONFIG).cat_temperature == 1.3
    assert [runs[i].task.name for i in (0, 12, 24)] == ["humaneval", "commonsense_qa", "truthful_qa"]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.dataset_split == "eval"
        assert config.cfg_scale == 0 and config.llada_reference_cfg
    humaneval = cli.build_config(runs[0], DEFAULT_CONFIG)
    assert (humaneval.code_dataset, humaneval.gen_length, humaneval.code_n_shots) == ("humaneval", 512, 0)

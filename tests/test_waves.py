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

    assert len(runs) == 60
    assert [(run.task.name, run.tag) for run in runs[36:38]] == [
        ("truthful_qa", "dbs_a5_t1p3"),
        ("truthful_qa", "dbs_a20_t1p3"),
    ]
    assert (runs[43].task.name, runs[43].tag) == ("truthful_qa", "d5p4_w320_t1p3")
    assert [(run.task.name, run.tag) for run in (runs[44], runs[52], runs[-1])] == [
        ("humaneval", "dbs_a5_t1p3"),
        ("commonsense_qa", "dbs_a5_t1p3"),
        ("commonsense_qa", "d5p4_w320_t1p3"),
    ]
    assert cli.build_config(runs[43], DEFAULT_CONFIG).cat_temperature == 1.3
    assert [runs[i].task.name for i in (0, 12, 24)] == ["humaneval", "commonsense_qa", "truthful_qa"]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.dataset_split == "eval"
        assert config.cfg_scale == 0 and config.llada_reference_cfg
    humaneval = cli.build_config(runs[0], DEFAULT_CONFIG)
    assert (humaneval.code_dataset, humaneval.gen_length, humaneval.code_n_shots) == ("humaneval", 512, 0)


def test_p1_covers_gsm8k_mbpp_and_arc_on_eval_splits():
    runs = get_wave("p1")

    assert len(runs) == 60
    assert [runs[i].task.name for i in (0, 20, 40)] == ["gsm8k", "mbpp", "ai2_arc"]
    assert [run.tag for run in runs[:2]] == ["ind_t0p7_s0of4", "ind_t1_s0of4"]
    assert [run.tag for run in runs[20:60:20]] == ["ind_t0p7", "ind_t0p7"]
    assert [run.tag for run in runs[52:54]] == ["dbs_a5_t1p3", "dbs_a20_t1p3"]
    assert runs[-1].tag == "d5p4_w320_t1p3"
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.dataset_split == "eval"
        assert config.cfg_scale == 0 and config.llada_reference_cfg
        assert config.n_groups * config.group_size == run.task.pool_size
    gsm8k = cli.build_config(runs[0], DEFAULT_CONFIG)
    assert (gsm8k.qa_dataset_len, gsm8k.qa_num_shards, gsm8k.qa_shard_index) == (-1, 4, 0)
    assert (gsm8k.n_groups, gsm8k.group_size) == (16, 1)
    mbpp = cli.build_config(runs[20], DEFAULT_CONFIG)
    assert (mbpp.code_dataset, mbpp.code_n_shots, mbpp.gen_length) == ("mbpp", 4, 256)
    assert cli.build_config(runs[-1], DEFAULT_CONFIG).cat_temperature == 1.3


@pytest.mark.parametrize("wave", ["s1", "e1", "e2", "b1"])
def test_side_waves_build_valid_configs_with_two_seeds(wave):
    runs = get_wave(wave)
    seeds = {cli.build_config(run, DEFAULT_CONFIG).seed for run in runs}
    assert seeds == {0, 1}
    half = len(runs) // 2
    assert all(run.tag.split("_seed")[1].startswith("0") for run in runs[:half])
    assert all(run.tag.split("_seed")[1].startswith("1") for run in runs[half:])
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.dataset_shuffle_seed == 0
        assert config.n_groups * config.group_size == run.task.pool_size


def test_s1_maps_paper_cfg_levels_to_the_reference_convention():
    from d5p4.diffusion_llada import cfg_is_active

    runs = get_wave("s1")
    assert len(runs) == 112
    assert (runs[0].task.name, runs[0].tag) == ("gsm8k_cfg1p5", "ind_t1_seed0_s0of4")
    configs = {run.task.name: cli.build_config(run, DEFAULT_CONFIG) for run in runs}
    assert configs["gsm8k_cfg1p5"].cfg_scale == 0.5
    assert configs["truthful_qa_cfg3"].cfg_scale == 2.0
    assert all(cfg_is_active(c.cfg_scale, reference_convention=c.llada_reference_cfg) for c in configs.values())
    assert configs["gsm8k_cfg2p5"].qa_num_shards == 4


def test_e1_decodes_in_blocks_of_32():
    for run in get_wave("e1"):
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert (config.block_length, config.llada_steps, config.gen_length) == (32, 256, 256)


def test_e2_runs_dream_with_a_dev_grid_check():
    runs = get_wave("e2")
    assert len(runs) == 66
    assert [run.task.name for run in runs[:10]] == ["dream_gsm8k_dev"] * 9 + ["dream_gsm8k"]
    dev = cli.build_config(runs[0], DEFAULT_CONFIG)
    assert (dev.model, dev.dataset_split, dev.qa_dataset_len, dev.dream_steps) == ("dream", "dev", 200, 256)
    qa = cli.build_config(runs[32], DEFAULT_CONFIG)
    assert (runs[32].task.entry, qa.qa_dataset, qa.dream_steps) == ("single_run_dream.py", "truthful_qa", 128)


def test_b1_tempered_remasking_samples_over_every_masked_position():
    runs = get_wave("b1")
    by_tag = {(run.task.name, run.tag): cli.build_config(run, DEFAULT_CONFIG) for run in runs}
    stochastic = by_tag[("gsm8k_stochastic", "d5p4_w20_t1_seed0_s0of4")]
    assert (stochastic.remasking, stochastic.selection_temperature, stochastic.selection_candidates) == (
        "selection_temperature",
        0.1,
        2,
    )
    tempered = by_tag[("truthful_qa_tempered", "ind_t1_tau0p3_seed1")]
    assert (tempered.selection_temperature, tempered.selection_candidates) == (0.3, 0)
    assert by_tag[("truthful_qa_tempered", "ind_t1_seed1")].remasking == "low_confidence"


def test_scale_pilot_keeps_the_pool_per_shape_and_puts_heavy_pools_first():
    runs = get_wave("scale")
    pools = [run.task.pool_size for run in runs]
    assert pools == sorted(pools, reverse=True)
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.n_groups * config.group_size == run.task.pool_size
        assert (config.qa_dataset, config.qa_num_shards, config.qa_shard_index, config.seed) == ("truthful_qa", 4, 0, 0)
        assert config.dataset_split == "eval" and config.cfg_scale == 0
    by_task = {}
    for run in runs:
        by_task.setdefault(run.task.name, set()).add(run.method)
    assert all(methods == {"greedy_map", "diverse_beam", "baseline"} for methods in by_task.values())


def test_b2_pairs_tempered_search_with_b1_tempered_independent_runs():
    b1_configs = {
        run.tag: cli.build_config(run, DEFAULT_CONFIG) for run in get_wave("b1") if run.task.name == "gsm8k_tempered"
    }
    runs = get_wave("b2")
    assert [run.tag for run in runs] == [
        "dbs_a80_t1_tau0p1_seed0_s0of4",
        "d5p4_w20_t1_tau0p1_seed0_s0of4",
        "dbs_a80_t1_tau0p3_seed0_s0of4",
        "d5p4_w20_t1_tau0p3_seed0_s0of4",
    ]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        baseline = b1_configs[f"ind_t1_tau{run.tag.split('_tau')[1]}"]
        assert run.task.name == "gsm8k_tempered"
        for key in (
            "remasking",
            "selection_temperature",
            "selection_candidates",
            "seed",
            "qa_num_shards",
            "qa_shard_index",
            "dataset_split",
            "dataset_shuffle_seed",
            "cfg_scale",
            "gen_length",
            "cat_temperature",
        ):
            assert getattr(config, key) == getattr(baseline, key), key
        assert config.n_groups * config.group_size == baseline.n_groups * baseline.group_size


def test_b2_full_covers_shards_1_to_3_with_the_shard_0_settings():
    shard_0 = {run.tag: cli.build_config(run, DEFAULT_CONFIG) for run in get_wave("b1") + get_wave("b2")}
    runs = get_wave("b2_full")
    per_shard = [
        f"{method}_tau{tau}_seed0" for tau in ("0p1", "0p3") for method in ("ind_t1", "dbs_a80_t1", "d5p4_w20_t1")
    ]
    assert [run.tag for run in runs] == [f"{tag}_s{shard}of4" for shard in (1, 2, 3) for tag in per_shard]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        reference = shard_0[run.tag.rsplit("_s", 1)[0] + "_s0of4"]
        assert run.task.name == "gsm8k_tempered"
        assert config.qa_shard_index == int(run.tag.rsplit("_s", 1)[1].split("of")[0])
        assert {k: v for k, v in vars(config).items() if k not in {"qa_shard_index", "comment"}} == {
            k: v for k, v in vars(reference).items() if k not in {"qa_shard_index", "comment"}
        }


def test_a3_dumps_the_first_50_questions_of_p1_and_transfer_with_the_same_settings():
    reference = {
        (run.task.name, run.tag): cli.build_config(run, DEFAULT_CONFIG) for run in get_wave("p1") + get_wave("transfer")
    }
    runs = get_wave("a3")
    assert [(run.task.name, run.tag) for run in runs] == [
        ("gsm8k_k8w4", "d5p4_w20_t1_s0of4"),
        ("gsm8k", "d5p4_w20_t1_s0of4"),
        ("gsm8k", "ind_t1_s0of4"),
        ("truthful_qa", "d5p4_w20_t1"),
        ("truthful_qa", "ind_t1"),
    ]
    ignored = {"qa_dataset_len", "selection_dump", "selection_dump_every", "n_groups", "group_size", "batch_size"}
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.selection_dump and config.selection_dump_every == 1
        # 50 questions: GSM8K takes shard 0 of the first 200 rows, TruthfulQA the first 50.
        assert config.qa_dataset_len // config.qa_num_shards == 50
        base = reference[("gsm8k" if run.task.name == "gsm8k_k8w4" else run.task.name, run.tag)]
        assert {k: v for k, v in vars(config).items() if k not in ignored | {"comment"}} == {
            k: v for k, v in vars(base).items() if k not in ignored | {"comment"}
        }
    k8w4 = cli.build_config(runs[0], DEFAULT_CONFIG)
    assert (k8w4.n_groups, k8w4.group_size) == (8, 4)


def _config_without(config, keys):
    return {k: v for k, v in vars(config).items() if k not in {"comment", *keys}}


def _gsm8k_and_truthful_qa_references():
    """P1's GSM8K shard 0 and the transfer wave's full-split TruthfulQA runs, keyed by unsharded tag."""
    return {
        (run.task.name, run.tag.removesuffix("_s0of4")): cli.build_config(run, DEFAULT_CONFIG)
        for run in get_wave("p1") + get_wave("transfer")
        if run.task.name in {"gsm8k", "truthful_qa"}
    }


def test_a2_ablations_change_one_part_of_d5p4_against_the_p1_and_transfer_references():
    references = _gsm8k_and_truthful_qa_references()
    runs = get_wave("a2")
    per_task = ["d5p3_w20_t1", "d5p4_w20_t1_quniform", "d5p4_w20_t1_qexp", "d5p4_w20_t1_qzscore"]
    checks = ["d5p4_w0_t1", "rnd_t1"]
    assert [(run.task.name, run.tag.removesuffix("_s0of4")) for run in runs] == [
        *(("gsm8k", tag) for tag in per_task),
        *(("truthful_qa", tag) for tag in per_task),
        *(("truthful_qa", tag) for tag in checks),
        *(("gsm8k", tag) for tag in checks),
    ]
    varied = {
        "d5p3_w20_t1": ("d5p4_w20_t1", {"transversal": False}),
        "d5p4_w20_t1_quniform": ("d5p4_w20_t1", {"_score_normalization": "uniform"}),
        "d5p4_w20_t1_qexp": ("d5p4_w20_t1", {"_score_normalization": "exp"}),
        "d5p4_w20_t1_qzscore": ("d5p4_w20_t1", {"_score_normalization": "zscore"}),
        "d5p4_w0_t1": ("d5p4_w20_t1", {"_w_interaction": 0.0}),
        "rnd_t1": ("gbs_t1", {"method": "random"}),
    }
    sharding = {"qa_num_shards", "qa_shard_index"}
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert (config.qa_num_shards, config.qa_shard_index, config.seed) == (4, 0, 0)
        reference_tag, changes = varied[run.tag.removesuffix("_s0of4")]
        reference = references[(run.task.name, reference_tag)]
        assert {key: getattr(config, key) for key in changes} == changes
        assert _config_without(config, {*changes, *sharding}) == _config_without(reference, {*changes, *sharding})


def test_parallel_decoding_only_changes_the_number_of_steps():
    references = _gsm8k_and_truthful_qa_references()
    runs = get_wave("parallel")
    assert [run.task.name for run in runs[::3]] == ["gsm8k_tps2", "gsm8k_tps4", "truthful_qa_tps2", "truthful_qa_tps4"]
    assert {run.tag for run in runs} == {"ind_t1_s0of4", "dbs_a80_t1_s0of4", "d5p4_w20_t1_s0of4"}
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        base_task, tokens_per_step = run.task.name.rsplit("_tps", 1)
        reference = references[(base_task, run.tag.removesuffix("_s0of4"))]
        assert config.gen_length == reference.gen_length == config.block_length
        assert config.llada_steps == config.gen_length // int(tokens_per_step)
        # The selection window is indexed by step and covers every one of them.
        assert config.subsample_start == 0 and config.subsample_end >= config.llada_steps
        assert (config.qa_num_shards, config.qa_shard_index) == (4, 0)
        ignored = {"llada_steps", "guidance_end", "qa_num_shards", "qa_shard_index"}
        assert _config_without(config, ignored) == _config_without(reference, ignored)


def test_b6_seed1_repeats_the_b2_full_tau_0p3_runs_with_seed_1():
    seed_0 = {
        run.tag: cli.build_config(run, DEFAULT_CONFIG)
        for run in get_wave("b1") + get_wave("b2") + get_wave("b2_full")
        if run.task.name == "gsm8k_tempered"
    }
    runs = get_wave("b6_seed1")
    per_shard = ["ind_t1_tau0p3", "dbs_a80_t1_tau0p3", "d5p4_w20_t1_tau0p3"]
    expected = [f"{tag}_seed1_s{shard}of4" for shard in range(4) for tag in per_shard]
    expected.remove("ind_t1_tau0p3_seed1_s0of4")
    assert [run.tag for run in runs] == expected
    # The left-out run is B1's, with the settings of the rest of the seed.
    b1_run = cli.build_config(next(r for r in get_wave("b1") if r.tag == "ind_t1_tau0p3_seed1_s0of4"), DEFAULT_CONFIG)
    assert _config_without(b1_run, {"seed"}) == _config_without(seed_0["ind_t1_tau0p3_seed0_s0of4"], {"seed"})
    assert b1_run.seed == 1
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        reference = seed_0[run.tag.replace("_seed1_", "_seed0_")]
        assert config.seed == 1 and run.task.name == "gsm8k_tempered"
        assert _config_without(config, {"seed"}) == _config_without(reference, {"seed"})


def test_b6_sweep_extends_the_tau_0p3_points_on_shard_0():
    reference = {run.tag: cli.build_config(run, DEFAULT_CONFIG) for run in get_wave("b2")}
    runs = get_wave("b6_sweep")
    assert [run.tag for run in runs] == [
        "dbs_a20_t1_tau0p3_seed0_s0of4",
        "dbs_a320_t1_tau0p3_seed0_s0of4",
        "d5p4_w5_t1_tau0p3_seed0_s0of4",
        "d5p4_w80_t1_tau0p3_seed0_s0of4",
    ]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        if run.method == "diverse_beam":
            base, key = reference["dbs_a80_t1_tau0p3_seed0_s0of4"], "_diversity_alpha"
        else:
            base, key = reference["d5p4_w20_t1_tau0p3_seed0_s0of4"], "_w_interaction"
        assert _config_without(config, {key}) == _config_without(base, {key})


def test_scale_gsm8k_keeps_the_pool_per_shape_and_the_p1_profile():
    p1_runs = {run.tag: cli.build_config(run, DEFAULT_CONFIG) for run in get_wave("p1") if run.task.name == "gsm8k"}
    runs = get_wave("scale_gsm8k")
    pools = [run.task.pool_size for run in runs]
    assert pools == sorted(pools, reverse=True)
    assert [run.task.name for run in runs[::3]] == ["gsm8k_k8w8", "gsm8k_k16w4", "gsm8k_k4w8"]
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.n_groups * config.group_size == run.task.pool_size
        assert (config.qa_num_shards, config.qa_shard_index, config.seed) == (4, 0, 0)
        reference = p1_runs[run.tag.replace("_seed0_", "_")]
        ignored = {"n_groups", "group_size", "batch_size"}
        assert _config_without(config, ignored) == _config_without(reference, ignored)


def test_b5_check_decodes_e2_questions_with_the_dream_instruct_protocol():
    from d5p4.dream_math import _shard_indexed_rows

    runs = get_wave("b5")
    configs = [cli.build_config(run, DEFAULT_CONFIG) for run in runs]
    assert [(run.task.name, run.tag) for run in runs] == [
        ("dream_gsm8k_check_greedy", "ind_t0_s0of4"),
        ("dream_gsm8k_check", "ind_t0p1_s0of4"),
        ("dream_gsm8k_check", "ind_t0p3_s0of4"),
        ("dream_gsm8k_check", "ind_t0p5_s0of4"),
        ("dream_gsm8k_check", "ind_t1_s0of4"),
        ("dream_humaneval_check", "ind_t0p1"),
    ]
    for config in configs:
        assert (config.model, config.dream_alg, config.dream_alg_temp, config.dream_top_p) == (
            "dream",
            "entropy",
            0.0,
            0.9,
        )
        assert config.gen_length == config.dream_steps  # one token per step
        assert config.math_answer_extractor == "lm_eval_flexible"
        assert config.dataset_split == "eval"
    greedy, *sampled, code = configs
    assert (greedy.cat_temperature, greedy.n_groups, greedy.group_size) == (0.0, 1, 1)
    assert [(c.cat_temperature, c.n_groups) for c in sampled] == [(0.1, 4), (0.3, 4), (0.5, 4), (1.0, 4)]
    assert (greedy.qa_dataset_len, greedy.qa_num_shards, greedy.qa_shard_index, greedy.gen_length) == (200, 4, 0, 256)
    # Shard 0 of the first 200 questions is the first 50 questions of E2's shard 0 of the full test split.
    check_rows = [index for index, _ in _shard_indexed_rows(list(range(200)), shard_index=0, num_shards=4)]
    e2_rows = [index for index, _ in _shard_indexed_rows(list(range(1319)), shard_index=0, num_shards=4)]
    assert check_rows == e2_rows[:50]
    assert runs[-1].task.entry == "dream_code.py"
    assert (code.code_dataset, code.code_n_shots, code.gen_length, code.cat_temperature) == ("humaneval", 0, 768, 0.1)


def test_b5_dream_searches_at_the_official_temperature():
    runs = get_wave("b5_dream")
    assert len(runs) == 36
    assert [run.task.name for run in runs[:12]] == ["dream_humaneval"] * 12
    for run in runs:
        config = cli.build_config(run, DEFAULT_CONFIG)
        assert config.n_groups * config.group_size == run.task.pool_size
        if config.method != "baseline":
            assert config.cat_temperature == 0.1
        assert run.task.reference_tag == "ind_t0p1"
    gsm8k = cli.build_config(runs[12], DEFAULT_CONFIG)
    assert (gsm8k.qa_dataset_len, gsm8k.qa_num_shards, gsm8k.qa_shard_index, gsm8k.seed) == (-1, 4, 0, 0)
    assert {cli.build_config(run, DEFAULT_CONFIG).seed for run in runs[24:]} == {1}

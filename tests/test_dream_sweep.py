from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_sweep_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / ".scripts" / "sweep_dream_gsm8k.py"
    spec = importlib.util.spec_from_file_location("sweep_dream_gsm8k", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_dream_sweep_has_equal_budget_method_seed_grid() -> None:
    sweep = _load_sweep_module()
    args = sweep._build_parser().parse_args([])

    entries = sweep.build_entries(args)

    assert len(entries) == 12
    assert [(entry.overrides["seed"], entry.overrides["method"]) for entry in entries] == [
        (seed, method)
        for seed in (0, 1, 2)
        for method in ("baseline", "greedy_map", "diverse_beam", "greedy_beam")
    ]
    assert {
        entry.overrides["n_groups"] * entry.overrides["group_size"] for entry in entries
    } == {16}

    for entry in entries:
        overrides = entry.overrides
        assert overrides["qa_dataset"] == "gsm8k"
        assert overrides["qa_n_shots"] == 0
        assert overrides["gen_length"] == 256
        assert overrides["dream_steps"] == 256
        assert overrides["dream_alg"] == "entropy"
        assert overrides["dream_alg_temp"] == 0.0
        assert overrides["cat_temperature"] == 1.0
        assert overrides["dream_top_p"] == 0.9
        assert overrides["resume_runs"] is True

    by_method = {entry.overrides["method"]: entry.overrides for entry in entries[:4]}
    assert by_method["baseline"]["n_groups"] == 16
    assert by_method["baseline"]["group_size"] == 1
    assert by_method["greedy_map"]["_w_interaction"] == 25.0
    assert by_method["diverse_beam"]["_diversity_alpha"] == 12.0
    assert all("--nproc_per_node=gpu" in entry.cmd for entry in entries)


def test_dream_sweep_filters_grid_and_forwards_cluster_paths() -> None:
    sweep = _load_sweep_module()
    args = sweep._build_parser().parse_args(
        [
            "--qa_dataset_len=1",
            "--seeds",
            "7",
            "--methods",
            "baseline",
            "greedy_map",
            "--compile_model=false",
            "--dream_model_path=/models/dream",
            "--dream_tokenizer=/models/dream",
            "--results_dir=/results",
            "--resume_db_dir=/results/resume",
            "--cache_dir=/cache",
        ],
    )

    entries = sweep.build_entries(args)

    assert len(entries) == 2
    for entry in entries:
        assert entry.overrides["qa_dataset_len"] == 1
        assert entry.overrides["seed"] == 7
        assert entry.overrides["compile_model"] is False
        assert entry.overrides["dream_model_path"] == "/models/dream"
        assert entry.overrides["resume_db_dir"] == "/results/resume"
        assert "qa_dataset_len=1" in entry.cmd
        assert "gen_length=256" in entry.cmd


def test_dream_sweep_gpu_count_does_not_change_semantic_config() -> None:
    sweep = _load_sweep_module()
    common = ["--qa_dataset_len=1", "--seeds", "0", "--methods", "greedy_map"]
    one_gpu = sweep.build_entries(sweep._build_parser().parse_args([*common, "--nproc=1"]))[0]
    two_gpu = sweep.build_entries(sweep._build_parser().parse_args([*common, "--nproc=2"]))[0]

    assert one_gpu.overrides == two_gpu.overrides
    assert "--nproc_per_node=1" in one_gpu.cmd
    assert "--nproc_per_node=2" in two_gpu.cmd

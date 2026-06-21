#!/usr/bin/env python3
"""
Orchestrates the LLaDA paper code generation sweep over 48 configurations:
  - 2 Datasets (HumanEval, MBPP)
  - 2 Remasking Methods (low_confidence, random)
  - 4 Sampling Methods (independent/baseline, greedy_map, diverse_beam, greedy_beam)
  - 3 Seeds (0, 1, 2)

TWO-PHASE PIPELINE DESIGN:
---------------------------
Phase 1: Generation Only (Fast)
  Run the sweep with `--skip_eval=true --resume_db_keep_completed=true`.
  All configurations will generate samples at maximum speed, caching generations
  directly to the resume SQLite DB without executing any evaluation tests.

Phase 2: Post-Evaluation
  Run the sweep with `--skip_eval=false --resume_db_keep_completed=true`.
  The script opens the exact same resume SQLite database (since 'skip_eval' is
  excluded from the semantic config hash). Because all generations are already
  completed in the DB, model sampling is skipped entirely. The script runs the
  code validation test suite on the cached generations, computes overall metrics,
  and writes the final JSON result files.

SKIP-IF-COMPLETE BEHAVIOR:
---------------------------
`llada_code.py` checks at startup (`is_run_completed_distributed`) if the target
run is already finalized (status = "complete") in the SQLite DB. If so, it prints
a skip notice and exits with status 0. The orchestrator then immediately moves
on to the next command.

COOPERATIVE WORKERS:
--------------------
Multiple one-GPU sweep workers may traverse the same command list against the
same resume DB directory. If a worker reaches an experiment whose DB lock is
already held by another live worker, `llada_code.py` exits cleanly and this
orchestrator moves on to the next configuration.

PORT ALLOCATION:
----------------
A new, unused master port is dynamically allocated for `torchrun` on each
subprocess spawn to prevent port collision issues.
"""

import argparse
import os
import socket
import subprocess
import sys


def main():  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser(description="Run sweep over LLaDA code benchmarks.")
    parser.add_argument(
        "--skip_eval",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Skip evaluation during generation (default: true)",
    )
    parser.add_argument(
        "--resume_db_keep_completed",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Keep completed resume databases (default: true)",
    )
    parser.add_argument(
        "--nproc",
        type=str,
        default="gpu",
        help="Number of GPUs / processes per node for torchrun (default: gpu)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print the commands, don't run them")
    args = parser.parse_args()

    datasets = ["humaneval", "mbpp"]
    remasking_methods = ["low_confidence", "random"]
    seeds = [0, 1, 2]

    # Define sampling methods
    # Each method has unique parameters depending on the dataset
    methods = ["baseline", "greedy_map", "diverse_beam", "greedy_beam"]

    commands = []

    for dataset in datasets:
        # Dataset-specific lengths/shots
        if dataset == "humaneval":
            gen_len = 512
            n_shots = 0
            subsample_end = 256
        else:
            gen_len = 256
            n_shots = 4
            subsample_end = 128

        for remasking in remasking_methods:
            for seed in seeds:
                for method in methods:
                    cmd_args = [
                        "torchrun",
                        f"--nproc_per_node={args.nproc}",
                        "llada_code.py",
                        "--config=_default.yaml",
                        "minimal_log=true",
                        "model=llada",
                        f"code_dataset={dataset}",
                        f"code_n_shots={n_shots}",
                        f"seed={seed}",
                        f"remasking={remasking}",
                        "logits_eos_inf=False",
                        "cfg_scale=1.0",
                        f"llada_steps={gen_len}",
                        f"gen_length={gen_len}",
                        f"block_length={gen_len}",
                        "confidence_eos_eot_inf=True",
                        f"skip_eval={args.skip_eval.lower()}",
                        f"resume_db_keep_completed={args.resume_db_keep_completed.lower()}",
                        "resume_runs=True",
                        f"method={method}",
                    ]

                    # Add method-specific parameters
                    if method == "baseline":
                        cmd_args.extend(
                            [
                                "n_groups=9",
                                "group_size=1",
                            ],
                        )
                    elif method == "greedy_map":
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                f"subsample_end={subsample_end}",
                                "_w_interaction=10.0",
                            ],
                        )
                    elif method == "diverse_beam":
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                f"subsample_end={subsample_end}",
                                "_diversity_alpha=20.0",
                            ],
                        )
                    elif method == "greedy_beam":
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                f"subsample_end={subsample_end}",
                            ],
                        )

                    cmd_args.append(
                        f"comment=LLaDA sweep: dataset={dataset}, remasking={remasking}, "
                        f"seed={seed}, method={method}, skip_eval={args.skip_eval}",
                    )
                    commands.append(cmd_args)

    print(f"Generated {len(commands)} commands for the sweep.")

    # We must run from src/d5p4 where llada_code.py lives
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/d5p4"))

    for idx, cmd in enumerate(commands):
        print("\n================================================================================")
        print(f"Running command {idx + 1}/{len(commands)}:")
        print(" ".join(cmd))
        print("================================================================================")

        if args.dry_run:
            continue

        # We need a new master port for torchrun to avoid port collisions
        # Find an open port

        s = socket.socket()
        s.bind(("", 0))
        master_port = str(s.getsockname()[1])
        s.close()

        # Inject master_port
        # We find where torchrun is, and inject --master_port there
        for i, val in enumerate(cmd):
            if val == "torchrun" or (val == "-m" and cmd[i + 1] == "torchrun"):
                idx_to_insert = i + 2 if val == "-m" else i + 1
                cmd.insert(idx_to_insert, f"--master_port={master_port}")
                break

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            sys.exit(e.returncode)

    # Run post-hoc Best-of-N selection for independent baseline runs when evaluation is enabled
    if args.skip_eval.lower() == "false":
        print("\n================================================================================")
        print("Running post-hoc Best-of-N selection for baseline (independent) runs...")
        print("================================================================================")

        bon_cmd = [
            "python3",
            "_oversample_baseline.py",
            "subsample_k=3",
            "resume_runs=True",
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OVERSAMPLE_BASELINE_PATH"] = os.path.join(cwd, "results")
        env["OVERSAMPLE_BASELINE_METHOD"] = "baseline"

        print("Running baseline post-processor:")
        print(" ".join(bon_cmd))

        if not args.dry_run:
            try:
                subprocess.run(bon_cmd, cwd=cwd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running baseline post-processing: {e}")
                sys.exit(e.returncode)


if __name__ == "__main__":
    main()

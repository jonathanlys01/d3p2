# Experiment waves on Jean Zay (`ejh@h100`)

A wave is a fixed list of generation runs defined in `src/d5p4/waves/` (`m0.py` is the first). `submit.sh` launches
one Slurm array task per run, then an evaluation job that starts once every array task has ended. The evaluation
job runs the existing `.scripts_next/evaluate_jz_*` scripts and writes a `summary.csv`.

## Layout on `$WORK_ejh`

| Path | Contents |
| --- | --- |
| `projects/d3p2/` | this repository and its `.venv` |
| `projects/d3p2_waves/<wave>/results/<task>/<tag>/` | generation JSONs, `.wave_run` (command, git revision), `.wave_done` |
| `projects/d3p2_waves/<wave>/resume/<task>/<tag>/` | resume databases; an interrupted run continues where it stopped |
| `projects/d3p2_waves/<wave>/eval/<task>/` | compact metric JSONs from the evaluators |
| `projects/d3p2_waves/<wave>/summary.csv` | one row per (run, selection metric) |
| `models/`, `datasets/` | assets from `download_assets.sh` |

Results live outside the repository, so updating the code never touches them. Llama-3-8B (perplexity), GPT-2
large, and bert-base-uncased are read from `$DSDIR/HuggingFace_Models`.

## One-time setup (login node)

```bash
git clone git@github.com:jonathanlys01/d3p2.git "$WORK_ejh/projects/d3p2"   # or `git pull` if it exists
cd "$WORK_ejh/projects/d3p2"
.jz_waves/download_assets.sh
```

`download_assets.sh` creates the `.venv` with `uv sync --frozen` if needed (uv in `~/.local/bin`), downloads
LLaDA-8B-Instruct, the Jina embeddings, and the six benchmark repositories, checks the `$DSDIR` models, and ends
with an offline preflight (`python -m d5p4.waves check`) that builds every run's config and loads each dev split.

## Running a wave

```bash
.jz_waves/submit.sh --smoke m0          # every run on 2 questions, dev QoS, root m0_smoke
.jz_waves/submit.sh m0                  # the real wave
python -m d5p4.waves status --wave m0 --root "$WORK_ejh/projects/d3p2_waves/m0"
.jz_waves/submit.sh --missing m0        # resubmit failed or timed-out runs, chaining a new evaluation
.jz_waves/submit.sh --eval-only m0      # re-evaluate without generating
```

The evaluation job refuses to run on an incomplete wave and prints the missing array indices; `--allow-partial`
overrides that. Generation is resumable: a resubmitted run continues from its resume database, and a run with a
`.wave_done` marker exits immediately. Array indices follow the run order of the wave definition, so do not reorder
a wave after submitting it; append new runs instead.

`submit.sh --dry-run` prints the `sbatch` commands without submitting. `python -m d5p4.waves list --wave m0`
shows each index, and `python -m d5p4.waves run --wave m0 --root R --config C --index I --dry-run` prints the
exact generation command.

## Adding a wave

Define its tasks and runs in a new module under `src/d5p4/waves/` with the helpers in `spec.py`, register it in
`src/d5p4/waves/__init__.py`, and add a test next to `tests/test_waves.py`. Each task names its generation script
(`llada_math.py`, `single_run_llada.py`, `llada_code.py`) and evaluation family (`math`, `qa`, `code`).

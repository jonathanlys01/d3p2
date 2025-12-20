"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from dataclasses import asdict

from config import Config
from exps.common import _bcast, print, run_experiment, run_sweep


SWEEP_NAME = "d3p2_cat_optuna_study"


def _objective(trial, og_config: Config):
    cat_temperature = trial.suggest_float("cat_temperature", 0.7, 1.1)

    dict_config = asdict(og_config)
    dict_config["cat_temperature"] = cat_temperature
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: cat_temperature={cat_temperature}")

    metrics = run_experiment(config)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    init_trials = [{"cat_temperature": qual} for qual in [0.7, 0.9, 1.1]]
    run_sweep(SWEEP_NAME, og_config, _objective, n_trials=200, init_trials=init_trials)

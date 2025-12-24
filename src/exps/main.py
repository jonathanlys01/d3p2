"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from dataclasses import asdict

from common_exps import _bcast, print, run_experiment, run_sweep
from config import Config


SWEEP_NAME = "d3p2_optuna_study"


def _objective(trial, og_config: Config):
    w_interaction = trial.suggest_float("w_interaction", 0.0, 8.0)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: w_inter={w_interaction}")

    metrics = run_experiment(config)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    init_trials = [{"w_interaction": qual} for qual in [0.0, 0.1, 0.3, 1.0, 3.0]]
    run_sweep(SWEEP_NAME, og_config, _objective, n_trials=200, init_trials=init_trials)

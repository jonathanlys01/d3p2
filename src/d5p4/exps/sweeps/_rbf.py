"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from dataclasses import asdict

import numpy as np

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d3p2_rbf_optuna_study"


def _objective(trial, og_config: Config, model, evaluator):
    w_interaction = trial.suggest_float("w_interaction", 0.0, 8.0)
    det_temperature = trial.suggest_float("determinant_temperature", 1e-5, 1.0, log=True)
    rbf_gamma = trial.suggest_float("rbf_gamma", 1e-2, 1e2, log=True)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["_temperature"] = det_temperature
    dict_config["_rbf_gamma"] = rbf_gamma
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: w_inter={w_interaction}, det_temp={det_temperature}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    init_trials = []
    for qual in np.linspace(0.0, 8.0, 3):
        for temp in [1e-5, 3e-3, 1.0]:
            for gamma in [1e-2, 1.0, 1e2]:
                init_trials.append(
                    {
                        "w_interaction": qual,
                        "determinant_temperature": temp,
                        "rbf_gamma": gamma,
                    },
                )

    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)

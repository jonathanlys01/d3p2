"""Optuna sweep for UDLM greedy MAP interaction weight."""

from dataclasses import asdict

import numpy as np

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "udlm_map_optuna_study"


def _objective(trial, og_config: Config, model, evaluator):
    w_interaction = trial.suggest_float("_w_interaction", 1e-1, 5e3, log=True)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)
    _bcast(config)

    print(f"Trial {trial.number}: _w_interaction={w_interaction}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    assert og_config.model == "udlm"
    assert og_config.method == "greedy_map"
    init_trials = [{"_w_interaction": value} for value in np.logspace(-1, 3, 5).tolist()]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)

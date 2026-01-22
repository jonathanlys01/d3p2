"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from dataclasses import asdict

import numpy as np

from common_exps import _bcast, print, run_experiment, run_sweep
from config import Config


SWEEP_NAME = "d3p2_divbs_optuna_study"


def _objective(trial, og_config: Config, model, evaluator):
    div_alpha = trial.suggest_float("_diversity_alpha", 0.0, 10.0)

    dict_config = asdict(og_config)
    dict_config["_diversity_alpha"] = div_alpha
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: _diversity_alpha={div_alpha}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    assert og_config.method == "diverse_beam"
    init_trials = [{"_diversity_alpha": alpha} for alpha in np.linspace(0.0, 10.0, 5).tolist()]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)

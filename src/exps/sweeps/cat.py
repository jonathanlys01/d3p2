"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

from dataclasses import asdict

from common_exps import _bcast, print, run_experiment, run_sweep
from config import Config


SWEEP_NAME = "d3p2_cat_optuna_study"


def _objective(trial, og_config: Config, model, evaluator):
    cat_temperature = trial.suggest_float("cat_temperature", 1.0, 1.5)

    dict_config = asdict(og_config)
    dict_config["cat_temperature"] = cat_temperature
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: cat_temperature={cat_temperature}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    init_trials = [{"cat_temperature": qual} for qual in [1.0, 1.25, 1.5]]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)

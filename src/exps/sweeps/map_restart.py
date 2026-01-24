"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
Manually restart a study from a previous run.
"""

from dataclasses import asdict

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from common_exps import _bcast, print, run_experiment, run_sweep
from config import Config


OLD_SWEEP_NAME = "MAIN_MAP_study"
SWEEP_NAME = "LOG_MAP_study"


def _objective(trial, og_config: Config, model, evaluator):
    # w_interaction = trial.suggest_float("w_interaction", 0.0, 8.0)
    w_interaction = trial.suggest_float("w_interaction", 1e-1, 5e3, log=True)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: w_inter={w_interaction}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()

    storage_old = JournalStorage(JournalFileBackend(f"optuna_{OLD_SWEEP_NAME}.log"))
    study_old = optuna.load_study(study_name=OLD_SWEEP_NAME, storage=storage_old)

    storage_new = JournalStorage(JournalFileBackend(f"optuna_{SWEEP_NAME}.log"))
    study_new = optuna.create_study(
        study_name=SWEEP_NAME,
        storage=storage_new,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    if len(study_new.trials) == 0:
        print("Migrating trials from old study...")
        trials = [
            trial
            for trial in study_old.trials
            if trial.params["w_interaction"] > 0 and trial.state == optuna.trial.TrialState.COMPLETE
        ]
        study_new.add_trials(trials)
        print(f"Successfully migrated {len(trials)} trials.")

        print("Enqueueing initial trials...")
        init_trials = [{"w_interaction": w} for w in np.logspace(-1, 3, 5).tolist()]
        for trial in init_trials:
            study_new.enqueue_trial(trial)
    else:
        print("New study already contains data. Skipping migration to avoid duplicates.")

    run_sweep(SWEEP_NAME, og_config, _objective, study_to_restart=study_new)

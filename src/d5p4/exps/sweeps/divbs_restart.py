"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
Manually restart a study from a previous run.
"""

from dataclasses import asdict

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


OLD_SWEEP_NAME = "d3p2_divbs_optuna_study"
SWEEP_NAME = "LOG_divbs_study"


def _objective(trial, og_config: Config, model, evaluator):
    # div_alpha = trial.suggest_float("_diversity_alpha", 0, 10)
    div_alpha = trial.suggest_float("_diversity_alpha", 1e-1, 5e3, log=True)

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
        # Create trials with the correct log-scale distribution
        log_dist = FloatDistribution(1e-1, 5e3, log=True)

        migrated_count = 0
        for old_trial in study_old.trials:
            alpha = old_trial.params["_diversity_alpha"]
            # Skip trials outside the new log-scale range or incomplete
            if alpha < 1e-1 or old_trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            new_trial = optuna.create_trial(
                params={"_diversity_alpha": alpha},
                distributions={"_diversity_alpha": log_dist},
                values=old_trial.values,
                user_attrs=old_trial.user_attrs,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study_new.add_trial(new_trial)
            migrated_count += 1

        print(f"Successfully migrated {migrated_count} trials.")

        print("Enqueueing initial trials...")
        init_trials = [{"_diversity_alpha": alpha} for alpha in np.logspace(-1, 3, 5).tolist()]
        for trial in init_trials:
            study_new.enqueue_trial(trial)
    else:
        print("New study already contains data. Skipping migration to avoid duplicates.")

    run_sweep(SWEEP_NAME, og_config, _objective, study_to_restart=study_new)

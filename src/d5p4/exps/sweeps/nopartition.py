"""
Optuna sweep seeded with the interaction values used by the brain MDLM runs.
"""

import os
from dataclasses import asdict

import optuna
from optuna.distributions import FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d3p2_np_restart_optuna_study"
BOOTSTRAP_TRIALS = (
    {"w_interaction": 1.0, "values": [39.99, 1.0]},
    {"w_interaction": 2.0, "values": [39.35, 1.0]},
    {"w_interaction": 4.0, "values": [40.33, 0.9991]},
    {"w_interaction": 8.0, "values": [41.91, 0.9592]},
    {"w_interaction": 8.32, "values": [43.40596734140976, 0.9567859709262848]},
    {"w_interaction": 8.64, "values": [44.252644671159935, 0.943466357588768]},
    {"w_interaction": 8.96, "values": [43.08892788490178, 0.9415103743970394]},
    {"w_interaction": 9.28, "values": [44.87565182954087, 0.9279617962241172]},
    {"w_interaction": 9.6, "values": [45.31131234506262, 0.9292161354422569]},
    {"w_interaction": 9.92, "values": [46.4665680492981, 0.9141208055615425]},
    {"w_interaction": 10.24, "values": [46.62024359044385, 0.9120342713594437]},
    {"w_interaction": 10.56, "values": [47.47364214060177, 0.8980215181410313]},
    {"w_interaction": 10.879999999999999, "values": [47.72794090827429, 0.8919494514167309]},
)

INTERACTION_MIN = min(t["w_interaction"] for t in BOOTSTRAP_TRIALS)
INTERACTION_MAX = max(t["w_interaction"] for t in BOOTSTRAP_TRIALS)
assert isinstance(INTERACTION_MIN, float) and isinstance(INTERACTION_MAX, float)


def _bootstrap_completed_trials(study: optuna.Study) -> None:
    dist = FloatDistribution(INTERACTION_MIN, INTERACTION_MAX * 1.1)
    for trial_data in BOOTSTRAP_TRIALS:
        w_interaction = trial_data["w_interaction"]
        values = trial_data["values"]
        assert isinstance(values, list) and len(values) == 2
        study.add_trial(
            optuna.create_trial(
                params={"w_interaction": w_interaction},
                distributions={"w_interaction": dist},
                values=values,
                user_attrs={"metrics": trial_data["metrics"]},
                state=optuna.trial.TrialState.COMPLETE,
            ),
        )


def _objective(trial, og_config: Config, model, evaluator):
    w_interaction = trial.suggest_float("w_interaction", INTERACTION_MIN, INTERACTION_MAX)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)
    _bcast(config)

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
    storage_path = f"optuna_{SWEEP_NAME}.log"
    db_exists = os.path.exists(storage_path)
    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.create_study(
        study_name=SWEEP_NAME,
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    if not db_exists and len(study.trials) == 0:
        print("Bootstrapping study with completed nopartition trials...")
        study.set_user_attr("og_config", asdict(og_config))
        _bootstrap_completed_trials(study)
    elif len(study.trials) == 0:
        study.set_user_attr("og_config", asdict(og_config))

    run_sweep(SWEEP_NAME, og_config, _objective, study_to_restart=study)

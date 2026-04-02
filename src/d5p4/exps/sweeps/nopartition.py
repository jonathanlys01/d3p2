"""
Optuna sweep seeded with the interaction values used by the brain MDLM runs.
"""

import os
from dataclasses import asdict

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d3p2_nopartition_optuna_study"
BASE_INTERACTION_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
INTERACTION_MIN = min(BASE_INTERACTION_VALUES)
INTERACTION_MAX = max(BASE_INTERACTION_VALUES)
N_INTERIOR_POINTS = 24
BOOTSTRAP_TRIALS = (
    {
        "w_interaction": 1.0,
        "values": [39.99, 1.0],
        "metrics": {
            "perplexity": 39.99,
            "perplexity_ci95_lower": 39.37,
            "perplexity_ci95_upper": 40.62,
            "corpus_perplexity": 39.98,
            "cosine_similarity": 1.0,
            "cosine_similarity_ci95": 7.176e-06,
            "distinct_2": 0.2096,
            "distinct_2_ci95": 0.001271,
            "empirical_entropy": 5.252,
            "empirical_entropy_ci95": 0.02124,
            "batch_empirical_entropy": 6.733,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 39.99 [39.37, 40.62] | C-PPL: 39.98 | CosSim: 1 pm 7.176e-06 | "
                "Dist-2: 0.2096 pm 0.001271 | Ent: 5.252 pm 0.02124 | B-Ent: 6.733 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
    {
        "w_interaction": 2.0,
        "values": [39.35, 1.0],
        "metrics": {
            "perplexity": 39.35,
            "perplexity_ci95_lower": 38.71,
            "perplexity_ci95_upper": 40.0,
            "corpus_perplexity": 39.35,
            "cosine_similarity": 1.0,
            "cosine_similarity_ci95": 3.793e-06,
            "distinct_2": 0.2089,
            "distinct_2_ci95": 0.001417,
            "empirical_entropy": 5.245,
            "empirical_entropy_ci95": 0.02303,
            "batch_empirical_entropy": 6.764,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 39.35 [38.71, 40] | C-PPL: 39.35 | CosSim: 1 pm 3.793e-06 | "
                "Dist-2: 0.2089 pm 0.001417 | Ent: 5.245 pm 0.02303 | B-Ent: 6.764 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
    {
        "w_interaction": 4.0,
        "values": [40.33, 0.9991],
        "metrics": {
            "perplexity": 40.33,
            "perplexity_ci95_lower": 39.67,
            "perplexity_ci95_upper": 41.0,
            "corpus_perplexity": 40.32,
            "cosine_similarity": 0.9991,
            "cosine_similarity_ci95": 0.001084,
            "distinct_2": 0.2107,
            "distinct_2_ci95": 0.002399,
            "empirical_entropy": 5.244,
            "empirical_entropy_ci95": 0.03119,
            "batch_empirical_entropy": 6.791,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 40.33 [39.67, 41] | C-PPL: 40.32 | CosSim: 0.9991 pm 0.001084 | "
                "Dist-2: 0.2107 pm 0.002399 | Ent: 5.244 pm 0.03119 | B-Ent: 6.791 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
    {
        "w_interaction": 8.0,
        "values": [41.91, 0.9592],
        "metrics": {
            "perplexity": 41.91,
            "perplexity_ci95_lower": 41.18,
            "perplexity_ci95_upper": 42.65,
            "corpus_perplexity": 41.9,
            "cosine_similarity": 0.9592,
            "cosine_similarity_ci95": 0.006631,
            "distinct_2": 0.2685,
            "distinct_2_ci95": 0.009504,
            "empirical_entropy": 5.27,
            "empirical_entropy_ci95": 0.02014,
            "batch_empirical_entropy": 6.791,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 41.91 [41.18, 42.65] | C-PPL: 41.9 | CosSim: 0.9592 pm 0.006631 | "
                "Dist-2: 0.2685 pm 0.009504 | Ent: 5.27 pm 0.02014 | B-Ent: 6.791 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
    {
        "w_interaction": 16.0,
        "values": [52.35, 0.8547],
        "metrics": {
            "perplexity": 52.35,
            "perplexity_ci95_lower": 51.48,
            "perplexity_ci95_upper": 53.24,
            "corpus_perplexity": 52.34,
            "cosine_similarity": 0.8547,
            "cosine_similarity_ci95": 0.003762,
            "distinct_2": 0.4164,
            "distinct_2_ci95": 0.003543,
            "empirical_entropy": 5.351,
            "empirical_entropy_ci95": 0.01606,
            "batch_empirical_entropy": 6.945,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 52.35 [51.48, 53.24] | C-PPL: 52.34 | CosSim: 0.8547 pm 0.003762 | "
                "Dist-2: 0.4164 pm 0.003543 | Ent: 5.351 pm 0.01606 | B-Ent: 6.945 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
    {
        "w_interaction": 32.0,
        "values": [55.69, 0.8372],
        "metrics": {
            "perplexity": 55.69,
            "perplexity_ci95_lower": 54.82,
            "perplexity_ci95_upper": 56.57,
            "corpus_perplexity": 55.67,
            "cosine_similarity": 0.8372,
            "cosine_similarity_ci95": 0.002566,
            "distinct_2": 0.4252,
            "distinct_2_ci95": 0.001384,
            "empirical_entropy": 5.389,
            "empirical_entropy_ci95": 0.01126,
            "batch_empirical_entropy": 7.056,
            "self_bleu": 100.0,
            "self_bleu_ci95": 2.789e-15,
            "metrics_summary": (
                "PPL: 55.69 [54.82, 56.57] | C-PPL: 55.67 | CosSim: 0.8372 pm 0.002566 | "
                "Dist-2: 0.4252 pm 0.001384 | Ent: 5.389 pm 0.01126 | B-Ent: 7.056 | "
                "S-BLEU: 100 pm 2.789e-15"
            ),
        },
    },
)


def _seed_interaction_values() -> list[float]:
    dense_region = np.linspace(8.0, 16.0, N_INTERIOR_POINTS + 2)[1:-1].tolist()
    return [*BASE_INTERACTION_VALUES[:4], *dense_region, *BASE_INTERACTION_VALUES[4:]]


def _bootstrap_completed_trials(study: optuna.Study) -> None:
    dist = FloatDistribution(INTERACTION_MIN, INTERACTION_MAX)
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


def _enqueue_seed_trials(study: optuna.Study) -> None:
    completed_values = {trial.params["w_interaction"] for trial in study.trials if "w_interaction" in trial.params}
    for value in _seed_interaction_values():
        if value not in completed_values:
            study.enqueue_trial({"w_interaction": value})


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
        _enqueue_seed_trials(study)
    elif len(study.trials) == 0:
        study.set_user_attr("og_config", asdict(og_config))
        _enqueue_seed_trials(study)

    run_sweep(SWEEP_NAME, og_config, _objective, study_to_restart=study)

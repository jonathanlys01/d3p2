"""
Shared experiment logic for distributed runs and Optuna sweeps.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

import idr_torch
import optuna
import torch
import torch.distributed as dist
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from config import RESULTS_DIR, Config
from diffusion_mdlm import MDLMSampler
from eval_core import Evaluator
from utils import compile_model, print, seed_all


def _bcast(obj):
    """Broadcast a single Python object from rank 0; return it on all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    is_master: bool = idr_torch.is_master  # type: ignore
    obj_list = [obj] if is_master else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _save(text, config, uid):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)


def generate_samples_with_model(config: Config, model: MDLMSampler):
    """Generate samples using a pre-initialized model."""
    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}, n_runs: {config.n_runs}")

    for _ in range(config.n_runs):
        samples = model.sample()
        texts.append(model.tokenizer.batch_decode(samples, skip_special_tokens=True))
        _save(texts, config, unique_id)

    samples = {
        "text_samples": texts,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }
    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    if master:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    return unique_id, master


def generate_samples(config: Config):
    """Generate samples by creating a new model instance."""
    model = MDLMSampler(config)
    model.model = compile_model(model.model, config)
    return generate_samples_with_model(config, model)


def eval_samples(unique_id: str, config: Config):
    evaluator = Evaluator(
        batch_size=16,
        force=True,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    metrics = {}
    # Evaluation expects the result file to exist
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(f"{unique_id}.json"):
            file_path = os.path.join(RESULTS_DIR, file)
            metrics = evaluator.eval_from_file(file_path)

    return metrics


def run_experiment(config: Config, model: MDLMSampler | None = None):
    """Run experiment with optional pre-initialized model."""
    torch.cuda.empty_cache()
    if model is None:
        unique_id, master = generate_samples(config)
    else:
        unique_id, master = generate_samples_with_model(config, model)
    if not master:
        return None
    metrics = eval_samples(str(unique_id), config)
    return metrics


def run_sweep(sweep_name, og_config, objective_fn, n_trials=None, init_trials=None):
    """
    Unified Optuna sweep loop handling both master and worker ranks.
    Model is initialized once and reused across all trials.
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.world_size,  # type: ignore
        rank=idr_torch.rank,  # type: ignore
    )

    device = f"cuda:{idr_torch.local_rank}"
    torch.cuda.set_device(device)

    is_master: bool = idr_torch.is_master  # type: ignore

    # Initialize model once before the sweep
    model = MDLMSampler(og_config)
    model.model = compile_model(model.model, og_config)

    if is_master:
        storage = JournalStorage(JournalFileBackend(f"optuna_{sweep_name}.log"))
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name=sweep_name,
            storage=storage,
            load_if_exists=True,
        )

        if len(study.trials) == 0:  # enqueue initial points
            study.set_user_attr("og_config", asdict(og_config))
            if init_trials:
                for trial_params in init_trials:
                    study.enqueue_trial(trial_params)

        study.optimize(lambda trial: objective_fn(trial, og_config, model), n_trials=n_trials)
        _bcast(False)  # signal workers to stop

    else:
        while True:
            proceed = _bcast(None)
            if not proceed:
                break

            cfg = _bcast(None)
            assert cfg is not None
            run_experiment(cfg, model)

    dist.destroy_process_group()

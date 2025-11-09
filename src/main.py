"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
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

from config import RESULTS_DIR, Config
from diffusion_mdlm import MDLMSampler
from eval_core import Evaluator
from utils import compile_model, print, seed_all


def _bcast(obj):
    """Broadcast a single Python object from rank 0; return it on all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    obj_list = [obj] if idr_torch.is_master else [None]
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


def generate_samples(config: Config):
    model = MDLMSampler(config)
    model.model = compile_model(model.model, config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}, n_runs: {config.n_runs}")

    for _ in range(config.n_runs):
        samples = model.sample(num_steps=config.num_steps)
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


def eval_samples(unique_id: str, config: Config):
    evaluator = Evaluator(
        batch_size=16,
        force=True,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    metrics = {}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(f"{unique_id}.json"):
            file_path = os.path.join(RESULTS_DIR, file)
            metrics = evaluator.eval_from_file(file_path)

    return metrics


def main(config: Config):
    unique_id, master = generate_samples(config)
    if not master:
        return None
    metrics = eval_samples(unique_id, config)
    return metrics


def _objective(trial: optuna.Trial, og_config: Config):
    w_interaction = trial.suggest_float("w_interaction", 0.0, 5.0)
    det_temperature = trial.suggest_float("determinant_temperature", 0.1, 2.0)

    dict_config = asdict(og_config)
    dict_config["w_interaction"] = w_interaction
    dict_config["determinant_temperature"] = det_temperature
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: w_interaction={w_interaction}, det_temp={det_temperature}")

    metrics = main(config)

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    print(f"Trial {trial.number} completed: Perplexity={perplexity}, Cosine Similarity={cos_sim}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.world_size,
        rank=idr_torch.rank,
    )

    device = f"cuda:{idr_torch.local_rank}"
    torch.cuda.set_device(device)

    is_master = idr_torch.is_master

    if is_master:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name="d3p2_optuna_study",
            storage="sqlite:///optuna_d3p2.db",
            load_if_exists=True,
        )

        if len(study.trials) == 0:
            study.enqueue_trial(
                {
                    "w_interaction": og_config.w_interaction,
                    "determinant_temperature": og_config.determinant_temperature,
                },
            )

        study.optimize(lambda trial: _objective(trial, og_config), n_trials=100)
        _bcast(False)

        dist.destroy_process_group()

    else:
        while True:
            proceed = _bcast(None)
            if not proceed:
                break

            cfg = _bcast(None)
            main(cfg)

        dist.destroy_process_group()

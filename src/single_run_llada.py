"""
Single run script for MDLM text generation.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from config import RESULTS_DIR, Config
from data import get_qa_dataset
from diffusion_llada import LLADASampler
from utils import compile_model, print, seed_all


def save(text, config, uid, rank=0):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    prompts: list[str] = [row.question for row in dataset][:limit]  # type: ignore

    for i in range(min(config.n_runs, len(prompts))):
        print(f"Sampling batch {i + 1}/{config.n_runs}...")
        samples = model.sample(prompt=prompts[i])
        texts_ = model.tokenizer.batch_decode(samples, skip_special_tokens=True)
        texts.append(texts_)
        save(texts, config, unique_id, rank=offset)

    samples = {
        "text_samples": texts,  # list of lists of strings
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }

    if model.distributed_utils is None or model.distributed_utils.rank == 0:  # save on master only (or non-distributed)
        postfix = str(uuid.uuid4())[:8]
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{postfix}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

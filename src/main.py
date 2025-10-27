"""
Main 5D3P experiment script.
(Distributed DPP Sampling for Discrete Diffusion Models)
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from config import RESULTS_DIR, Config
from diffusion import DDM
from utils import print, seed_all


def main():
    config = Config()

    model = DDM(config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    for i in range(config.n_runs):
        print(f"Sampling batch {i + 1}/{config.n_runs}...")
        samples = model.sample(num_steps=config.num_steps)
        texts.append(model.tokenizer.batch_decode(samples, skip_special_tokens=True))

    samples = {
        "text_samples": texts,  # list of lists of strings
        "config": asdict(config),
    }

    if model.distributed_utils is None or model.distributed_utils.rank == 0:  # save on master only (or non-distributed)
        postfix = str(uuid.uuid4())[:8]
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{postfix}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

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
from diffusion_mdlm import MDLMSampler
from utils import compile_model, print, seed_all


def save(text, config, uid):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)

        def concat_samples_same_config_postfix(results_dir=RESULTS_DIR, write_out=True):
            """
            Find all temp_*.json in results_dir, group by (config, postfix) where postfix is the trailing
            part after the last underscore in the filename, and concatenate their text_samples.
            Returns a dict keyed by (config_json_str, postfix) -> {"text_samples": [...], "config": {...}}.
            If write_out is True, writes a concat_<postfix>.json file for each group.
            """
            groups = {}
            for fname in os.listdir(results_dir):
                if not fname.startswith("temp_") or not fname.endswith(".json"):
                    continue
                path = os.path.join(results_dir, fname)
                with open(path, "r") as f:
                    data = json.load(f)

                name_no_ext = fname[:-5]
                postfix = name_no_ext.rsplit("_", 1)[1] if "_" in name_no_ext else ""

                config_obj = data.get("config", {})
                config_key = json.dumps(config_obj, sort_keys=True)
                key = (config_key, postfix)

                groups.setdefault(key, {"text_samples": [], "config": config_obj})
                groups[key]["text_samples"].extend(data.get("text_samples", []))

            if write_out:
                os.makedirs(results_dir, exist_ok=True)
                for (_, postfix), group in groups.items():
                    safe_postfix = str(postfix).replace(os.sep, "_")
                    out_name = f"concat_{safe_postfix}.json"
                    with open(os.path.join(results_dir, out_name), "w") as f:
                        json.dump(group, f, indent=4)

            return groups


def main():
    config = Config()

    model = MDLMSampler(config)
    model.model = compile_model(model.model, config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    for i in range(config.n_runs):
        print(f"Sampling batch {i + 1}/{config.n_runs}...")
        samples = model.sample(num_steps=config.num_steps)
        texts.append(model.tokenizer.batch_decode(samples, skip_special_tokens=True))
        save(texts, config, unique_id)

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
        if file.startswith("temp_") and file.endswith(f"{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

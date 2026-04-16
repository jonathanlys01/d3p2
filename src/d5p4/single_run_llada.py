"""
Single run script for MDLM text generation.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import Evaluator
from d5p4.utils import compile_model, print, seed_all


def save(text, config, uid, rank=0, references=None):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }
    if references is not None:
        samples["references"] = references

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore
    references_all: list[list[str]] = [row.correct_answers for row in rows]  # type: ignore

    for i, prompt in enumerate(prompts):
        print(f"Sampling batch {i + 1}/{len(prompts)}...", progress=True)
        samples = model.sample(prompt=prompt)
        texts_ = []
        for sample in samples:
            prompt_tokens = model._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            texts_.append(gen_text)

        texts.append(texts_)
        save(texts, config, unique_id, rank=offset, references=references_all[: i + 1])

    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    metrics = None
    if master:
        print("Running evaluation...")
        evaluator = Evaluator(
            batch_size=config.eval_batch_size,
            force=True,
            ppl_model_id=config.ppl_model_id,
            cos_model_id=config.cos_model_id,
        )
        metrics = evaluator.evaluate(texts, references=references_all)
        assert metrics["metrics_summary"] is not None
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    samples = {
        "text_samples": texts,
        "references": references_all,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }
    if metrics is not None:
        samples["metrics"] = metrics

    if master:  # save on master only (or non-distributed)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(config.results_dir, exist_ok=True)
        output_path = os.path.join(config.results_dir, f"exp-{name}.json")
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved in {output_path}")

    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

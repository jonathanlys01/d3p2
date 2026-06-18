"""Single run script for GIDD question answering."""

import json
import os
import uuid
from datetime import datetime
from typing import Any

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_gidd import GIDDSampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.utils import compile_model, print, seed_all


def _decode_generations(model: GIDDSampler, prompt: str, raw_samples: Any) -> list[str]:
    prompt_tokens = model._preprocess_prompt(prompt)
    prompt_len = prompt_tokens.shape[1]
    generations: list[str] = []
    for sample in raw_samples:
        completion_tokens = sample[prompt_len:]
        gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
        generations.append(gen_text)
    return generations


def save(text, config, uid, rank=0, references=None):
    samples = build_generation_result_payload(
        text_samples=text,
        config=config,
        references=references,
        experiment_id=str(uid),
    )
    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()
    model = GIDDSampler(config)
    if config.posterior_sampler != "gidd_hf_generate":
        model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []
    unique_id = uuid.uuid4()
    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    print(f"Experiment ID: {unique_id}")

    if config.prompt is not None:
        prompts = [config.prompt]
        references_all = None
    else:
        dataset = get_qa_dataset(config)
        limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
        rows = list(dataset.itertuples())[:limit]
        prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
        references_all: list[list[str]] | None = [row.correct_answers for row in rows]  # type: ignore[union-attr]

    for i, prompt in enumerate(prompts):
        print(f"Sampling batch {i + 1}/{len(prompts)}...")
        samples = model.sample(prompt=prompt)
        texts.append(_decode_generations(model, prompt, samples))
        references = references_all[: i + 1] if references_all is not None else None
        save(texts, config, unique_id, rank=offset, references=references)

    metrics = None
    if master:
        if references_all is None:
            print("Skipping evaluation because prompt mode has no dataset references.")
        elif config.skip_eval:
            print("Skipping evaluation because skip_eval=True.")
        else:
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

    payload = build_generation_result_payload(
        text_samples=texts,
        config=config,
        references=references_all,
        metrics=metrics,
        experiment_id=str(unique_id),
    )

    if master:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(config.results_dir, exist_ok=True)
        output_path = os.path.join(config.results_dir, f"exp-{name}.json")
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved in {output_path}")

    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

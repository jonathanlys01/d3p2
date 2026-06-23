"""Single run script for GIDD question answering."""

import json
import os
from datetime import datetime
from typing import Any

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_gidd import GIDDSampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import (
    prepare_resumable_run,
    release_resumable_run,
    sync_resume_item,
)
from d5p4.utils import compile_model, print, seed_all


def _decode_generations(model: GIDDSampler, prompt: str, raw_samples: Any) -> list[str]:
    prompt_tokens = model._preprocess_prompt(prompt)
    prompt_len = prompt_tokens.shape[1]
    generations: list[str] = []
    for sample in raw_samples:
        completion_tokens = sample[prompt_len:]
        gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
        generations.append(gen_text)
        print(f"  Generated: {gen_text}")
    return generations


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()

    if config.prompt is not None:
        prompts = [config.prompt]
        references_all = None
    else:
        dataset = get_qa_dataset(config)
        limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
        rows = list(dataset.itertuples())[:limit]
        prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
        references_all: list[list[str]] | None = [row.correct_answers for row in rows]  # type: ignore[union-attr]

    workflow_id = "prompt_generation:gidd"
    preflight = prepare_resumable_run(
        config=config,
        workflow_id=workflow_id,
        prompts=prompts,
        references=references_all,
        prefix="prompt",
        mode="prompt_generation",
    )
    if preflight.should_exit:
        return

    offset = preflight.offset
    master = preflight.master
    seed_all(config.seed + offset)

    model = GIDDSampler(config)
    if config.posterior_sampler != "gidd_hf_generate":
        model.model = compile_model(model.model, config, dynamic=True)

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    work_items = preflight.work_items
    texts = []

    if master:
        print(f"Experiment ID: {unique_id}")

    try:
        for i, prompt_item in enumerate(prompts):
            prompt = sync_resume_item(prompt_item, model.distributed_utils)
            if i in completed_indices:
                if not master:
                    continue
                assert store is not None
                generation = store.get_generation(i)
                assert generation is not None
                raw_samples = generation["tokens"]
                decoded = generation["decoded"] or _decode_generations(model, prompt, raw_samples)
                if generation["decoded"] is None:
                    store.record_decoded(item_index=i, decoded=decoded)
            else:
                if master:
                    print(f"Sampling {i + 1}/{len(prompts)}...", progress=True)
                raw_samples = model.sample(prompt=prompt)
                if not master:
                    continue
                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=model._preprocess_prompt(prompt).shape[1],
                    )
                decoded = _decode_generations(model, prompt, raw_samples)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded)

            if master:
                texts.append(decoded)
    finally:
        if store is not None:
            store.close()

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    metrics = None
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

    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
    os.makedirs(config.results_dir, exist_ok=True)
    output_path = os.path.join(config.results_dir, f"exp-{name}.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved in {output_path}")
    release_resumable_run(config=config, workflow_id=workflow_id, work_items=work_items, result_path=output_path)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

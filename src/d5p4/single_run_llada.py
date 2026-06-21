"""
Single run script for LLaDA text generation.
"""

import json
import os
from datetime import datetime

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import (
    release_resumable_run,
    run_generator_loop,
)
from d5p4.utils import compile_model, print, seed_all


LLADA_INTERNAL_SCORE_METADATA = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}


def _select_group_representatives(
    texts: list[str],
    scores: list[float],
    group_size: int,
) -> tuple[list[str], list[int]]:
    if len(texts) != len(scores):
        raise ValueError(f"Expected one score per generated sequence, got {len(scores)} scores for {len(texts)} texts.")
    if group_size <= 1:
        return texts.copy(), list(range(len(texts)))
    if len(texts) % group_size != 0:
        raise ValueError(
            f"Expected generated sequence count divisible by group_size, got {len(texts)} and {group_size}.",
        )

    selected_texts = []
    selected_indices = []
    for start in range(0, len(texts), group_size):
        group_scores = scores[start : start + group_size]
        best_local_idx = max(range(group_size), key=lambda idx: group_scores[idx])
        best_idx = start + best_local_idx
        selected_texts.append(texts[best_idx])
        selected_indices.append(best_idx)

    return selected_texts, selected_indices


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    use_internal_representatives = config.group_size > 1
    master = model.distributed_utils is None or model.distributed_utils.rank == 0

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore
    references_all: list[list[str]] = [row.correct_answers for row in rows]  # type: ignore

    workflow_id = "prompt_generation:llada"
    if master:
        print("Dumping final-step internal confidence scores for every generated sequence.")
    if use_internal_representatives and master:
        print("Using final-step internal scores to select one evaluation representative per group.")

    def sample_fn(prompt: str):
        return model.sample(prompt=prompt, return_internal_scores=True)

    def decode_fn(prompt: str, tokens) -> list[str]:
        prompt_len = model._preprocess_prompt(prompt).shape[1]
        generations = []
        for sample in tokens:
            completion_tokens = sample[prompt_len:]
            generations.append(model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip())
        return generations

    def post_process_fn(
        _i: int,
        _prompt: str,
        generations: list[str],
        scores: list[float] | None,
    ) -> tuple[list[str] | None, list[int] | None]:
        if use_internal_representatives and scores is not None:
            return _select_group_representatives(generations, scores, config.group_size)
        return None, None

    loop_outputs = run_generator_loop(
        config=config,
        model=model,
        prompts=prompts,
        references=references_all,
        workflow_id=workflow_id,
        prefix="prompt",
        mode="prompt_generation",
        sample_fn=sample_fn,
        decode_fn=decode_fn,
        post_process_fn=post_process_fn,
    )

    if loop_outputs.get("claimed_by_another_worker"):
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    texts = loop_outputs["generations"]
    internal_scores_all = loop_outputs["internal_scores"] or []
    eval_texts = loop_outputs["eval_generations"]
    eval_selected_indices = loop_outputs["selected_indices"]
    unique_id = loop_outputs["unique_id"]
    work_items = loop_outputs["work_items"]

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    metrics = None
    if config.skip_eval:
        print("Skipping evaluation because skip_eval=True.")
    else:
        print("Running evaluation...")
        evaluator = Evaluator(
            batch_size=config.eval_batch_size,
            force=True,
            ppl_model_id=config.ppl_model_id,
            cos_model_id=config.cos_model_id,
        )
        metric_texts = eval_texts if use_internal_representatives else texts
        metrics = evaluator.evaluate(metric_texts, references=references_all)
        assert metrics["metrics_summary"] is not None
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    eval_selection = None
    if use_internal_representatives:
        eval_selection = {
            "method": "final_internal_signal",
            "score_key": "internal_scores",
            "score_method": LLADA_INTERNAL_SCORE_METADATA["method"],
            "group_size": config.group_size,
        }

    samples = build_generation_result_payload(
        text_samples=texts,
        config=config,
        references=references_all,
        eval_text_samples=eval_texts if use_internal_representatives else None,
        internal_scores=internal_scores_all if internal_scores_all else None,
        eval_selected_indices=eval_selected_indices if use_internal_representatives else None,
        eval_selection=eval_selection,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores_all else None,
        metrics=metrics,
        experiment_id=str(unique_id),
    )

    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
    os.makedirs(config.results_dir, exist_ok=True)
    output_path = os.path.join(config.results_dir, f"exp-{name}.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
    print(f"Saved in {output_path}")
    release_resumable_run(config=config, workflow_id=workflow_id, work_items=work_items, result_path=output_path)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

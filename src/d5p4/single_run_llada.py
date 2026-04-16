"""
Single run script for LLaDA text generation.
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


def save(  # noqa: PLR0913
    text,
    config,
    uid,
    rank=0,
    references=None,
    eval_text=None,
    eval_internal_scores=None,
    eval_selected_indices=None,
):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }
    if references is not None:
        samples["references"] = references
    if eval_text is not None:
        samples["eval_text_samples"] = eval_text
    if eval_internal_scores is not None:
        samples["eval_internal_scores"] = eval_internal_scores
    if eval_selected_indices is not None:
        samples["eval_selected_indices"] = eval_selected_indices

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


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
    texts = []
    eval_texts = []
    eval_internal_scores = []
    eval_selected_indices = []
    use_internal_representatives = config.group_size > 1
    master = model.distributed_utils is None or model.distributed_utils.rank == 0

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")
    if use_internal_representatives and master:
        print("Using final-step internal scores to select one evaluation representative per group.")

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore
    references_all: list[list[str]] = [row.correct_answers for row in rows]  # type: ignore

    for i, prompt in enumerate(prompts):
        print(f"Sampling batch {i + 1}/{len(prompts)}...", progress=True)
        if use_internal_representatives:
            samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)
        else:
            samples = model.sample(prompt=prompt)
            internal_scores = None

        prompt_tokens = model._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        texts_ = []
        for sample in samples:
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            texts_.append(gen_text)

        texts.append(texts_)
        if use_internal_representatives and master:
            assert internal_scores is not None
            scores = [float(score) for score in internal_scores.detach().cpu().tolist()]
            eval_texts_, selected_indices = _select_group_representatives(texts_, scores, config.group_size)
            eval_texts.append(eval_texts_)
            eval_internal_scores.append(scores)
            eval_selected_indices.append(selected_indices)

        save(
            texts,
            config,
            unique_id,
            rank=offset,
            references=references_all[: i + 1],
            eval_text=eval_texts if use_internal_representatives and master else None,
            eval_internal_scores=eval_internal_scores if use_internal_representatives and master else None,
            eval_selected_indices=eval_selected_indices if use_internal_representatives and master else None,
        )

    metrics = None
    if master:
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

    samples = {
        "text_samples": texts,
        "references": references_all,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }
    if use_internal_representatives and master:
        samples["eval_text_samples"] = eval_texts
        samples["eval_internal_scores"] = eval_internal_scores
        samples["eval_selected_indices"] = eval_selected_indices
        samples["eval_selection"] = {
            "method": "final_internal_signal",
            "score_method": "final_step_mean_token_logprob",
            "group_size": config.group_size,
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

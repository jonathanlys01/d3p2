"""Single-run Dream prompt and QA generation."""

import json
import os
from datetime import datetime
from typing import Any, cast

import torch

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_dream import DreamSampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import prepare_resumable_run, release_resumable_run, sync_resume_item
from d5p4.utils import compile_model, print, seed_all


DREAM_INTERNAL_SCORE_METADATA = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}

# Increment this when sampling changes invalidate stored token generations.
# Version 5 masks LM-head ids that the tokenizer cannot decode.
DREAM_WORKFLOW_VERSION = 5


def _stop_token_ids(tokenizer: Any) -> set[int]:
    stop_ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        stop_ids.add(eos)
    elif eos is not None:
        stop_ids.update(int(token_id) for token_id in eos)

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        end_turn = convert("<|im_end|>")
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(end_turn, int) and end_turn >= 0 and end_turn != unk:
            stop_ids.add(end_turn)
    return stop_ids


def _safe_decode(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode *token_ids*, dropping any id the tokenizer cannot map to a token.

    Dream's LM head is wider than its tokenizer vocabulary, so an id sampled
    before the sampler learned to mask those columns — or replayed from an older
    resume database — decodes to ``None`` and blows up ``"".join(tokens)``.
    """
    try:
        return cast(str, tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        pass
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    kept = [token_id for token_id, token in zip(token_ids, tokens, strict=True) if isinstance(token, str)]
    dropped = len(token_ids) - len(kept)
    if dropped:
        print(f"Dropped {dropped}/{len(token_ids)} undecodable token ids while decoding.")
    return cast(str, tokenizer.decode(kept, skip_special_tokens=True))


def _decode_generations(
    model: DreamSampler,
    prompt: str,
    raw_samples: torch.Tensor,
    prompt_len: int | None = None,
) -> list[str]:
    if prompt_len is None:
        prompt_len = model._preprocess_prompt(prompt).shape[1]
    stop_ids = _stop_token_ids(model.tokenizer)
    generations = []
    for sample in raw_samples:
        completion = sample[prompt_len:].tolist()
        stop_positions = [idx for idx, token_id in enumerate(completion) if token_id in stop_ids]
        if stop_positions:
            completion = completion[: stop_positions[0]]
        decoded = _safe_decode(model.tokenizer, completion)
        generations.append(decoded.strip())
    return generations


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()
    assert config.model == "dream"

    if config.prompt is not None:
        prompts = [config.prompt]
        references_all = None
    else:
        dataset = get_qa_dataset(config)
        limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
        rows = list(dataset.itertuples())[:limit]
        prompts = [row.question for row in rows]  # type: ignore[union-attr]
        references_all = [row.correct_answers for row in rows]  # type: ignore[union-attr]

    workflow_id = f"prompt_generation:dream:v{DREAM_WORKFLOW_VERSION}"
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

    seed_all(config.seed + preflight.offset)
    model = DreamSampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    texts: list[list[str]] = []
    internal_scores_all: list[list[float]] = []

    if preflight.master:
        print(f"Experiment ID: {unique_id}")

    try:
        for i, prompt_item in enumerate(prompts):
            prompt = sync_resume_item(prompt_item, model.distributed_utils)
            if i in completed_indices:
                if not preflight.master:
                    continue
                assert store is not None
                generation = store.get_generation(i)
                assert generation is not None
                raw_samples = generation["tokens"]
                scores = generation["internal_scores"] or []
                decoded = generation["decoded"] or _decode_generations(
                    model,
                    prompt,
                    raw_samples,
                    generation["prompt_len"],
                )
                if generation["decoded"] is None:
                    store.record_decoded(item_index=i, decoded=decoded)
            else:
                if preflight.master:
                    print(f"Sampling {i + 1}/{len(prompts)}...")
                raw_samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)
                if not preflight.master:
                    continue
                prompt_len = raw_samples.shape[1] - config.gen_length
                scores = [float(score) for score in internal_scores.detach().cpu().tolist()]
                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=prompt_len,
                        internal_scores=scores,
                    )
                decoded = _decode_generations(model, prompt, raw_samples, prompt_len)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded)

            if preflight.master:
                texts.append(decoded)
                internal_scores_all.append(scores)
    finally:
        if store is not None:
            store.close()

    if not preflight.master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    metrics = None
    if references_all is None:
        print("Skipping evaluation because prompt mode has no dataset references.")
    elif config.skip_eval:
        print("Skipping evaluation because skip_eval=True.")
    else:
        evaluator = Evaluator(
            batch_size=config.eval_batch_size,
            force=True,
            ppl_model_id=config.ppl_model_id,
            cos_model_id=config.cos_model_id,
        )
        metrics = evaluator.evaluate(texts, references=references_all)
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    payload = build_generation_result_payload(
        text_samples=texts,
        config=config,
        references=references_all,
        internal_scores=internal_scores_all,
        internal_score_metadata=DREAM_INTERNAL_SCORE_METADATA,
        metrics=metrics,
        experiment_id=str(unique_id),
    )
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    os.makedirs(config.results_dir, exist_ok=True)
    output_path = os.path.join(config.results_dir, f"exp-{name}.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved in {output_path}")
    release_resumable_run(
        config=config,
        workflow_id=workflow_id,
        work_items=preflight.work_items,
        result_path=output_path,
    )

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()

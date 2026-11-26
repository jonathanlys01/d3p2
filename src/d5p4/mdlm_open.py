"""Open-ended MDLM generation on held-out OpenWebText prompts: one batch of n_groups * group_size sequences per prompt.

Each prompt is a held-out chunk (`d5p4.data.owt_heldout`): BOS and its first `open_prompt_tokens` tokens stay fixed
and MDLM fills the other 1024 - 1 - open_prompt_tokens positions in `mdlm_steps` denoising steps.

Search methods (Greedy and Diverse Beam Search, D5P4) select one candidate per group at every step and expand it into
group_size children. Their last step selects once more and copies each survivor group_size times, so the final batch
holds n_groups distinct lineages, each repeated: a run keeps the first row of every group, k = n_groups texts per
prompt (the number of groups whose copies differ is logged). Independent sampling (method=baseline, group_size 1) keeps
its whole pool; the front summaries score it as a random k of the pool.

Every prompt is generated under its own seed, derived from `seed` and the prompt index, so a resumed run reproduces
an uninterrupted one and the seeds of a run are independent replicates on the same prompts. Evaluation (GPT-2 Large
perplexity, MAUVE features, self-cosine, Self-BLEU) runs afterwards in `d5p4.waves.open_eval`.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import torch

from d5p4.config import Config
from d5p4.data.owt_heldout import load_open_chunks
from d5p4.data.splits import shard_rows
from d5p4.diffusion_mdlm import MDLMSampler
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import prepare_resumable_run, release_resumable_run
from d5p4.utils import compile_model, get_tokenizer, print, seed_all


WORKFLOW_ID = "open_generation:mdlm"
SEED_STRIDE = 1_000_003


def prompt_seed(seed: int, prompt_index: int) -> int:
    """Seed of one prompt: the same in every method, distinct across seeds and prompts."""
    return seed * SEED_STRIDE + prompt_index


def kept_rows(batch_size: int, group_size: int) -> list[int]:
    """Rows a run keeps: the first copy of each group (search), or the whole pool (group_size 1)."""
    return list(range(0, batch_size, group_size))


def copy_mismatches(tokens: torch.Tensor, group_size: int) -> int:
    """Groups whose final copies are not all identical (expected 0: the last step leaves nothing masked)."""
    if group_size <= 1:
        return 0
    grouped = tokens.view(-1, group_size, tokens.size(-1))
    return int((grouped != grouped[:, :1]).any(dim=-1).any(dim=-1).sum().item())


def initial_state(prefix: list[int], batch_size: int, length: int, mask_index: int) -> torch.Tensor:
    x = torch.full((batch_size, length), mask_index, dtype=torch.int64)
    x[:, : len(prefix)] = torch.tensor(prefix, dtype=torch.int64)
    return x


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():  # noqa: PLR0915
    config = Config()
    assert config.model == "mdlm", "mdlm_open.py generates with MDLM only"

    tokenizer = get_tokenizer(config, "mdlm")
    chunks = load_open_chunks(config, tokenizer)
    indexed = shard_rows(list(range(len(chunks.prompts))), config)
    dataset_indices = [index for index, _ in indexed]
    prefixes = [chunks.prefix(index) for index in dataset_indices]
    prompts = [tokenizer.decode(prefix[1:]) for prefix in prefixes]
    references = [
        [tokenizer.decode(chunks.continuation(chunks.prompts[index]), skip_special_tokens=True).strip()]
        for index in dataset_indices
    ]

    preflight = prepare_resumable_run(
        config=config,
        workflow_id=WORKFLOW_ID,
        prompts=prompts,
        references=references,
        prefix="owt",
        mode="prompt_generation",
    )
    if preflight.should_exit:
        return
    assert preflight.distributed_utils is None, "mdlm_open.py runs on one GPU; launch it without torchrun"
    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id

    sampler = MDLMSampler(config)
    sampler.model = compile_model(sampler.model, config)
    rows = kept_rows(config.batch_size, config.group_size)
    texts: list[list[str]] = []
    seconds: list[float | None] = []
    mismatches: list[int] = []
    leftover_masks: list[int] = []
    print(
        f"Experiment ID: {unique_id}; {len(prompts)} prompts, pool {config.batch_size}, keeping {len(rows)} rows, "
        f"{config.mdlm_steps} steps, prompts of {config.open_prompt_tokens} tokens",
    )

    try:
        for position, (index, prefix) in enumerate(zip(dataset_indices, prefixes, strict=True)):
            if position in completed:
                assert store is not None
                generation = store.get_generation(position)
                assert generation is not None
                tokens = generation["tokens"]
                elapsed = (generation["generation_metadata"] or {}).get("seconds")
            else:
                print(f"Sampling {position + 1}/{len(prompts)} (prompt {index})...", progress=True)
                seed_all(prompt_seed(config.seed, index))
                init_x = initial_state(prefix, config.batch_size, sampler.model_length, sampler.mask_index)
                _synchronize()
                started = time.time()
                tokens = sampler.sample(init_x=init_x).detach().cpu()
                _synchronize()
                elapsed = time.time() - started
                if store is not None:
                    store.record_generated(
                        item_index=position,
                        token_ids=tokens,
                        prompt_len=len(prefix),
                        generation_metadata={"seconds": elapsed, "prompt_index": index},
                    )
            expected_prefix = torch.tensor(prefix, dtype=tokens.dtype).expand(tokens.size(0), -1)
            assert torch.equal(tokens[:, : len(prefix)], expected_prefix), "the prompt changed during generation"
            mismatches.append(copy_mismatches(tokens, config.group_size))
            kept = tokens[rows, len(prefix) :]
            leftover_masks.append(int((kept == sampler.mask_index).sum().item()))
            texts.append([tokenizer.decode(row.tolist(), skip_special_tokens=True).strip() for row in kept])
            seconds.append(elapsed)
    finally:
        if store is not None:
            store.close()

    timed = [s for s in seconds if s is not None]
    if timed:
        print(f"Generation time per prompt: mean {sum(timed) / len(timed):.1f} s over {len(timed)} prompts")
    if any(mismatches):
        print(f"Warning: {sum(mismatches)} groups have final copies that differ; kept the first copy of each.")

    payload = build_generation_result_payload(
        text_samples=texts,
        config=config,
        references=references,
        experiment_id=str(unique_id),
        extra={
            "dataset_indices": dataset_indices,
            "prompts": prompts,
            "open_generation": {
                "prompt_tokens": config.open_prompt_tokens,
                "pool": config.batch_size,
                "kept_rows": rows,
                "generation_seconds": seconds,
                "copy_mismatch_groups": mismatches,
                "leftover_masks": leftover_masks,
            },
        },
    )
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    os.makedirs(config.results_dir, exist_ok=True)
    output_path = os.path.join(config.results_dir, f"exp-{name}.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Saved in {output_path}")
    release_resumable_run(
        config=config,
        workflow_id=WORKFLOW_ID,
        work_items=preflight.work_items,
        result_path=output_path,
    )


if __name__ == "__main__":
    main()

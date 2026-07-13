"""Command-line inference for the custom LLaDA sampler."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from config import D5P4Config
from sampler import generate_d5p4
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "jonathanlys01/LLaDA-8B-Instruct-D5P4"
DEFAULTS = D5P4Config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULTS.steps)
    parser.add_argument("--gen-length", type=int, default=DEFAULTS.gen_length)
    parser.add_argument("--block-length", type=int, default=DEFAULTS.block_length)
    parser.add_argument("--temperature", type=float, default=DEFAULTS.temperature)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULTS.cfg_scale)
    parser.add_argument("--remasking", choices=("low_confidence", "random"), default=DEFAULTS.remasking)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-groups", type=int, default=DEFAULTS.n_groups)
    parser.add_argument("--group-size", type=int, default=DEFAULTS.group_size)
    parser.add_argument("--resample-start", type=int, default=DEFAULTS.resample_start)
    parser.add_argument("--resample-end", type=int, default=DEFAULTS.resample_end)
    parser.add_argument("--kernel-type", choices=("cosine", "rbf"), default=DEFAULTS.kernel_type)
    parser.add_argument("--kernel-method", choices=("multiplicative", "additive"), default=DEFAULTS.kernel_method)
    parser.add_argument("--quality-weight", type=float, default=DEFAULTS.quality_weight)
    parser.add_argument("--rbf-gamma", type=float, default=DEFAULTS.rbf_gamma)
    parser.add_argument(
        "--score-method",
        choices=("entropy", "mean_token_confidence"),
        default=DEFAULTS.score_method,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = encoded.to(model.device)

    config = D5P4Config(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        n_groups=args.n_groups,
        group_size=args.group_size,
        resample_start=args.resample_start,
        resample_end=args.resample_end,
        kernel_type=args.kernel_type,
        kernel_method=args.kernel_method,
        quality_weight=args.quality_weight,
        rbf_gamma=args.rbf_gamma,
        score_method=args.score_method,
    )
    output_ids = generate_d5p4(
        model,
        encoded.input_ids,
        encoded.attention_mask,
        config=config,
    )
    prompt_length = encoded.input_ids.shape[1]
    completion_ids = output_ids[:, prompt_length:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    for completion in completions:
        print(completion)


if __name__ == "__main__":
    main()

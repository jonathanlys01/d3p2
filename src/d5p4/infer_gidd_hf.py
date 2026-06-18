"""Standalone Hugging Face inference for the configured GIDD checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from d5p4.config import Config
from d5p4.gidd_ref.modeling_gidd import GiddForDiffusionLM
from d5p4.utils import process_model_args, seed_all


DEFAULT_CONFIG_PATH = Path(__file__).with_name("_default.yaml")


@dataclass(frozen=True)
class InferenceArgs:
    config: Path
    model_path: str
    cache_dir: str
    prompt: list[str]
    prompt_file: Path | None
    num_samples: int
    max_length: int
    min_length: int
    steps: int
    block_length: int
    temperature: float
    top_p: float | None
    top_k: int | None
    sampling_method: Literal["ancestral", "adaptive"]
    noise_schedule: Literal["linear", "cosine"]
    tokens_per_step: int
    dtype: Literal["auto", "bfloat16", "float16", "float32"]
    device: str
    seed: int
    local_files_only: bool
    trust_remote_code: bool
    show_progress: bool
    include_prompt: bool
    json_output: bool


def _dtype(value: str) -> torch.dtype:
    if value == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _token_id(tokenizer: Any, attr: str, fallback: int) -> int:
    value = getattr(tokenizer, attr, None)
    return fallback if value is None else int(value)


def _load_default_config(path: Path) -> Config:
    base = OmegaConf.structured(Config(disable_sys_args=True))
    file_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(base, file_cfg, {"disable_sys_args": True, "model": "gidd"})
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    return Config(**cfg_dict)


def _build_parser(default_cfg: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config containing gidd_model_path",
    )
    parser.add_argument("--model-path", default=default_cfg.gidd_model_path, help="HF checkpoint path or model id")
    parser.add_argument("--cache-dir", default=default_cfg.cache_dir, help="Transformers cache directory")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to run. May be repeated. Defaults to one short smoke-test prompt.",
    )
    parser.add_argument("--prompt-file", type=Path, help="Optional text file with one prompt per line")
    parser.add_argument("--num-samples", type=int, default=default_cfg.n_groups, help="Samples per prompt")
    parser.add_argument("--max-length", type=int, default=default_cfg.gen_length, help="Generated tokens per prompt")
    parser.add_argument("--min-length", type=int, default=0, help="Minimum generated tokens before EOS is allowed")
    parser.add_argument("--steps", type=int, default=default_cfg.diffusion_steps, help="Denoising steps per block")
    parser.add_argument(
        "--block-length",
        type=int,
        default=default_cfg.gidd_block_length,
        help="GIDD generation block size",
    )
    parser.add_argument("--temperature", type=float, default=default_cfg.cat_temperature, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling cutoff")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff")
    parser.add_argument("--sampling-method", choices=["ancestral", "adaptive"], default="ancestral")
    parser.add_argument("--noise-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--tokens-per-step", type=int, default=1, help="Adaptive sampling tokens per step")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    parser.add_argument("--local-files-only", action="store_true", help="Disallow model/tokenizer downloads")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Decode the full sequence instead of completion only",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit machine-readable JSON")
    return parser


def _parse_args() -> InferenceArgs:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = bootstrap.parse_known_args()
    default_cfg = _load_default_config(known.config)
    parser = _build_parser(default_cfg)
    args = parser.parse_args()
    return InferenceArgs(**vars(args))


def _read_prompts(args: InferenceArgs) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        prompts.extend(line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip())
    if not prompts:
        prompts.append("Write a short answer: what is 2 + 2?")
    return prompts


def _load_model_and_tokenizer(args: InferenceArgs) -> tuple[GiddForDiffusionLM, Any]:
    local_files_only = args.local_files_only or os.path.isdir(args.model_path)
    tokenizer_args = process_model_args(args.model_path, cache_dir=args.cache_dir)
    model_args = process_model_args(args.model_path, cache_dir=args.cache_dir)
    tokenizer_args["local_files_only"] = local_files_only
    model_args["local_files_only"] = local_files_only

    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args, trust_remote_code=args.trust_remote_code)
    model = GiddForDiffusionLM.from_pretrained(
        **model_args,
        dtype=_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()
    return model, tokenizer


def _encode_prompt(tokenizer: Any, prompt: str, device: str, num_samples: int) -> torch.Tensor:
    encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (input_ids.numel() == 0 or input_ids[0, 0].item() != bos):
        bos_tensor = torch.full((input_ids.size(0), 1), int(bos), dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
    return input_ids.repeat(num_samples, 1)


def _decode_samples(
    tokenizer: Any,
    prompt_len: int,
    generated: torch.Tensor,
    *,
    include_prompt: bool,
) -> list[str]:
    decoded: list[str] = []
    for sample in generated:
        tokens = sample if include_prompt else sample[prompt_len:]
        decoded.append(tokenizer.decode(tokens.tolist(), skip_special_tokens=True).strip())
    return decoded


def main() -> None:
    args = _parse_args()
    seed_all(args.seed)
    prompts = _read_prompts(args)
    model, tokenizer = _load_model_and_tokenizer(args)

    params = asdict(args)
    params["config"] = str(params["config"])
    params["prompt_file"] = None if params["prompt_file"] is None else str(params["prompt_file"])

    results = []
    for prompt in prompts:
        input_ids = _encode_prompt(tokenizer, prompt, args.device, args.num_samples)
        generated = model.generate(
            inputs=input_ids,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            block_length=args.block_length,
            steps=args.steps,
            top_p=args.top_p,
            top_k=args.top_k,
            bos_token_id=_token_id(tokenizer, "bos_token_id", 0),
            eos_token_id=_token_id(tokenizer, "eos_token_id", 1),
            pad_token_id=_token_id(tokenizer, "pad_token_id", 2),
            mask_token_id=_token_id(tokenizer, "mask_token_id", 3),
            sampling_method=args.sampling_method,
            noise_schedule=args.noise_schedule,
            tokens_per_step=args.tokens_per_step,
            show_progress=args.show_progress,
        )
        generations = _decode_samples(
            tokenizer,
            input_ids.shape[1],
            generated.cpu(),
            include_prompt=args.include_prompt,
        )
        results.append({"prompt": prompt, "generations": generations})

    payload = {"parameters": params, "results": results}
    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("Parameters:")
    print(json.dumps(params, indent=2))
    for result in results:
        print(f"\nPrompt: {result['prompt']}")
        for idx, text in enumerate(result["generations"], start=1):
            print(f"[{idx}] {text}")


if __name__ == "__main__":
    main()

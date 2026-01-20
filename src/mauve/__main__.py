"""
Compute MAUVE score between a corpus of generated text (JSON from MDLM experiments)
and a reference corpus (stored as .bin file with tokenized text).
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from config import CACHE_DIR
from mauve.compute_mauve import compute_mauve
from utils import process_model_args


def load_reference_texts(
    bin_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 5000,
) -> list[str]:
    """
    Load reference texts from a .bin file.
    The .bin is a numpy memmap of uint16 tokens encoded with a transformers tokenizer.

    Args:
        bin_path: Path to the .bin file containing tokenized data
        tokenizer: Transformers tokenizer to use for decoding
        max_samples: Maximum number of sequences to load

    Returns:
        List of decoded text strings
    """
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eos_token_id = tokenizer.eos_token_id

    # Split on EOS tokens to get individual sequences
    texts = []
    current_tokens = []

    for token in arr:
        if token == eos_token_id:
            if current_tokens:
                text = tokenizer.decode(current_tokens, skip_special_tokens=False)
                if text.strip():  # Skip empty texts
                    texts.append(text)
                    if len(texts) >= max_samples:
                        break
            current_tokens = []
        else:
            current_tokens.append(int(token))

    # Handle last sequence if no trailing EOS
    if current_tokens and len(texts) < max_samples:
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if text.strip():
            texts.append(text)

    print(f"Loaded {len(texts)} reference texts from {bin_path}")
    return texts


def load_samples_from_json(json_path: str) -> list[str]:
    """
    Load generated samples from a JSON file (MDLM experiment format).
    Expected structure: {"text_samples": [[str, ...], ...], ...}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    text_samples = data.get("text_samples", [])

    # Flatten nested lists - text_samples is typically a list of lists
    flattened = []
    for item in text_samples:
        if isinstance(item, list):
            for text in item:
                if isinstance(text, str) and text.strip():
                    flattened.append(text.strip())
        elif isinstance(item, str) and item.strip():
            flattened.append(item.strip())

    print(f"Loaded {len(flattened)} samples from {json_path}")
    return flattened


def main():
    parser = argparse.ArgumentParser(description="Compute MAUVE score between reference and generated text.")
    parser.add_argument("bin_path", type=str, help="Path to the reference .bin file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with generated samples")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-large",
        help="Model to use for featurization (default: gpt2-large)",
    )
    parser.add_argument(
        "--reference_tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer used to encode the reference .bin file (default: gpt2)",
    )
    parser.add_argument(
        "--max_ref_samples",
        type=int,
        default=5000,
        help="Maximum number of reference samples to load (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for featurization (default: 8)",
    )
    args = parser.parse_args()

    # Initialize tokenizer for decoding reference .bin file
    ref_tokenizer_args = process_model_args(args.reference_tokenizer, cache_dir=CACHE_DIR)
    ref_tokenizer = AutoTokenizer.from_pretrained(**ref_tokenizer_args)

    # Load reference texts from .bin
    ref_texts = load_reference_texts(args.bin_path, tokenizer=ref_tokenizer, max_samples=args.max_ref_samples)

    # Load generated samples from JSON
    gen_texts = load_samples_from_json(args.json_path)

    if not ref_texts or not gen_texts:
        print("Error: No valid texts loaded from one or both sources.")
        return

    # Initialize model and tokenizer for MAUVE computation (following eval_core.py pattern)
    model_args = process_model_args(args.model, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(**model_args)
    tokenizer = AutoTokenizer.from_pretrained(**model_args)

    # Compute MAUVE
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Computing MAUVE with {len(ref_texts)} reference and {len(gen_texts)} generated texts...")

    result = compute_mauve(
        p_text=ref_texts,
        q_text=gen_texts,
        models=(model, tokenizer),
        device_id=device_id,
        batch_size=args.batch_size,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print(f"MAUVE Score: {result.mauve:.4f}")
    print(f"MAUVE* Score (smoothed): {result.mauve_star:.4f}")
    print(f"Frontier Integral: {result.frontier_integral:.4f}")
    print(f"Frontier Integral* (smoothed): {result.frontier_integral_star:.4f}")
    print(f"Number of Buckets: {result.num_buckets}")
    print("=" * 50)


if __name__ == "__main__":
    main()

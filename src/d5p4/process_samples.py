"""
Script to subsample text generations and re-compute metrics.
Deduplicates beams for map samples and selects best-F1 for independent samples.
"""

import argparse
import glob
import json
import os

from data.qa import get_qa_dataset

from d5p4.config import Config
from d5p4.eval_core import Evaluator


def process_file(file_path: str, evaluator: Evaluator, metric: str, k: int):  # noqa: C901
    print(f"Processing {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)

    # Check if already processed
    if "results_subsampled" in data and not args.force:
        print(f"Skipping {file_path} (already processed)")
        return

    cfg_dict = data.get("config", None)
    assert cfg_dict is not None

    cfg_dict["disable_sys_args"] = True

    cfg = Config(**cfg_dict)

    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    correct_answers = dataset["correct_answers"].tolist()
    # Get text samples
    # Support both dict (cfg_map/ref style) and list (normal/baseline style)
    text_samples_data = data.get("text_samples", {})

    # Normalize to a dict-like structure for processing
    # If list: {"default": samples}
    # If dict: samples
    if isinstance(text_samples_data, list):
        # Verify length matches references
        if len(text_samples_data) != len(correct_answers):
            print(f"Warning: len samples ({len(text_samples_data)}) != len refs ({len(correct_answers)})")
            # Proceed anyway for robust handling if possible, or just slice?
            # Let's rely on the loop below to zip safely or handle index error if accessed by index
            pass

        # Determine unique key for this single entry
        # Use filename or just "default"
        text_samples_map = {"default": text_samples_data}
    elif isinstance(text_samples_data, dict):
        text_samples_map = text_samples_data
    else:
        print(f"Unknown text_samples format in {file_path}")
        return

    results_subsampled = {}

    for cfg_val, samples in text_samples_map.items():
        if len(samples) != len(correct_answers):
            print(f"Warning: mismatch len samples ({len(samples)}) vs refs ({len(correct_answers)}) for {cfg_val}")
            continue

        subsampled_texts = []
        is_map = "cfg_map" in file_path

        if is_map:
            for beam_list in samples:
                unique_beams = sorted(set(beam_list))
                subsampled_texts.append(unique_beams)
        else:
            # Use evaluator.evaluate_baseline
            selected = evaluator.evaluate_baseline(
                full_sequences=samples,
                metric=metric,
                k=k,
                references=correct_answers,
            )
            # evaluate_baseline returns list of list of selected strings
            subsampled_texts = selected

        # Evaluate the subsampled texts
        metrics = evaluator.evaluate(subsampled_texts, references=correct_answers)
        results_subsampled[cfg_val] = metrics

    # Save results
    data["results_subsampled"] = results_subsampled

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Finished {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Syntax: --folders /path1 /path2 ... (nargs='+' gathers all following args into a list)
    parser.add_argument("--folders", nargs="+", required=True, help="Folders to process")
    parser.add_argument("--k", type=int, default=4, help="Number of samples to select")
    parser.add_argument("--ppl_model_id", type=str, default="gpt2")
    parser.add_argument("--cos_model_id", type=str, default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--force", action="store_true", help="Force re-processing")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Evaluator(
        batch_size=8,  # Auto
        force=args.force,
        ppl_model_id=args.ppl_model_id,
        cos_model_id=args.cos_model_id,
    )

    for folder in args.folders:
        files = glob.glob(os.path.join(folder, "*.json"))
        for f in files:
            process_file(f, evaluator, args.metric, args.k)

import json
import os
from dataclasses import fields

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.eval_core import Evaluator


def _select_and_evaluate_baseline(
    evaluator: Evaluator,
    texts: list[list[str]],
    metric: str,
    subsample_k: int,
    references: list[list[str]] | None = None,
) -> tuple[list[list[str]], dict[str, float | str]]:
    selected = evaluator.evaluate_baseline(texts, metric, subsample_k, references=references)
    metrics = evaluator.evaluate(selected, references=references)
    return selected, metrics


def _load_references(current_config: Config, expected_groups: int) -> list[list[str]] | None:
    if not current_config.qa_dataset:
        return None

    try:
        dataset = get_qa_dataset(current_config)
        limit = current_config.qa_dataset_len if current_config.qa_dataset_len > 0 else len(dataset)
        references = [row.correct_answers for row in dataset.itertuples()][:limit]
        print(f"Loaded {len(references)} references for {current_config.qa_dataset}")
    except Exception as e:
        print(f"Warning: Could not load references for {current_config.qa_dataset}: {e}")
        return None

    if len(references) != expected_groups:
        print(
            "Warning: Reference count does not match text groups "
            f"({len(references)} refs vs {expected_groups} groups); skipping reference-based metrics.",
        )
        return None

    return references


if __name__ == "__main__":
    config = Config()
    evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    path = os.path.expanduser("~/src/tries/2026-03-30-fixing-baseline")
    files = sorted([f for f in os.listdir(path) if f.endswith(".json") and "-bon-" not in f])

    subsample_k = config.subsample_k
    assert subsample_k != 0

    print("Using global subsample_k: ", subsample_k)

    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        # Try to load config from the result file
        file_config_dict = data.get("config", {})
        current_config = config
        if file_config_dict:
            # Create a new config object with values from the file
            # filter only valid fields
            valid_fields = {f.name for f in fields(Config)}
            filtered_config = {k: v for k, v in file_config_dict.items() if k in valid_fields}
            filtered_config.pop("disable_sys_args", None)
            current_config = Config(disable_sys_args=True, **filtered_config)

        texts = data["text_samples"]

        references = _load_references(current_config, len(texts))
        metrics_to_run = ["ppl", "f1"] if references is not None else ["ppl"]

        for metric in metrics_to_run:
            print(f"File: {file} | Metric: {metric}")
            selected, metrics = _select_and_evaluate_baseline(
                evaluator,
                texts,
                metric,
                subsample_k,
                references=references,
            )

            # Save dummy result file
            save_data = {
                "config": file_config_dict,
                "metrics": metrics,
                "text_samples": selected,
                "experiment_id": data.get("experiment_id", ""),
            }
            out_name = file.replace(".json", f"-bon-{metric}.json")
            with open(os.path.join(path, out_name), "w") as f_out:
                json.dump(save_data, f_out, indent=4)

            print("-" * 80)
            for key, value in metrics.items():
                print(f"{metric}_{key}: {value}")
            print("-" * 80)

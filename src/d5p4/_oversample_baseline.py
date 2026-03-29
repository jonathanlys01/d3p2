import json
import os

from d5p4.config import Config
from d5p4.eval_core import Evaluator


if __name__ == "__main__":
    config = Config()
    evaluator = Evaluator(
        batch_size=config.batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    path = os.path.expanduser("~/src/tries/2026-03-29-ppl-proxy")
    files = [f for f in os.listdir(path) if f.endswith(".json")]

    subsample_k = config.subsample_k

    assert subsample_k != 0

    print("Using subsample_k: ", subsample_k)

    for file in files:
        with open(os.path.join(path, file), "r") as f:
            data = json.load(f)

        texts = data["text_samples"]
        new_texts = []

        selected = evaluator.evaluate_baseline(texts, "ppl", subsample_k)
        metrics = evaluator.evaluate(selected)
        print(metrics)

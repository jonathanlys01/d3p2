import json

from config import Config
from eval_core import Evaluator


if __name__ == "__main__":
    config = Config()
    evaluator = Evaluator(
        batch_size=config.batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    path = "/Brain/private/j21lys/d3p2/src/results/exp-20260127_161300_dddd5af5-0f8a-42ed-a62c-ecf59c4614ec.json"
    bs = 8

    with open(path, "r") as f:
        data = json.load(f)

    texts = data["text_samples"]
    new_texts = []

    selected = evaluator.evaluate_baseline(texts, "ppl", bs)  # select top 8 per batch

    # repeat each element 8 times
    for i in range(len(selected)):
        new_texts.extend(selected[i] * bs)

    # rebatch
    new_texts = [new_texts[i : i + bs] for i in range(0, len(new_texts), bs)]

    # evaluate
    metrics = evaluator.evaluate(new_texts)
    print(metrics)

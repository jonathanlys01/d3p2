import pandas as pd
from datasets import load_dataset

from config import Config


def truthful_qa(cfg: Config) -> pd.DataFrame:
    dataset = load_dataset(cfg.truthful_qa_path, "generation", cache_dir=cfg.cache_dir)["validation"]
    dataset = dataset.shuffle(seed=cfg.seed)
    questions = [item["question"] for item in dataset]
    good = [item["correct_answers"] for item in dataset]
    bad = [item["incorrect_answers"] for item in dataset]

    df = pd.DataFrame({"question": questions, "correct_answers": good, "incorrect_answers": bad})
    return df


def commonsense_qa(cfg: Config) -> pd.DataFrame:
    dataset = load_dataset(cfg.commonsense_qa_path, cache_dir=cfg.cache_dir)["validation"]
    dataset = dataset.shuffle(seed=cfg.seed)
    questions = [item["question"] for item in dataset]
    good = []
    bad = []

    for item in dataset:
        answer_key = item["answerKey"]
        choices = item["choices"]["text"]
        good.append([choices[ord(answer_key) - ord("A")]])
        bad.append([choice for i, choice in enumerate(choices) if i != ord(answer_key) - ord("A")])

    df = pd.DataFrame({"question": questions, "correct_answers": good, "incorrect_answers": bad})

    return df


if __name__ == "__main__":
    for name, func in [("TruthfulQA", truthful_qa), ("CommonsenseQA", commonsense_qa)]:
        print(f"Loading {name} dataset...")
        df = func()
        print(df.head())
        print(f"Total samples: {len(df)}")

        # average number of correct answers
        avg_correct = df["correct_answers"].apply(len).mean()
        print(f"Average number of correct answers: {avg_correct}")

        # average number of incorrect answers
        avg_incorrect = df["incorrect_answers"].apply(len).mean()
        print(f"Average number of incorrect answers: {avg_incorrect}")

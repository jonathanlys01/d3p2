"""Inspect tokenized prompt lengths for the first questions in a QA dataset.

This mirrors the prompt preprocessing used by ``LLADASampler`` without loading
the model itself. It is intended for debugging late, data-dependent failures.

Example:
    ./venv/bin/python -m d5p4.check_llada_question_lengths model=llada qa_dataset=truthful_qa
"""

from __future__ import annotations

from dataclasses import dataclass

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.data.math_ds import gsm8k
from d5p4.utils import get_tokenizer


DEFAULT_LIMIT = 500
PREVIEW_CHARS = 100


@dataclass(frozen=True)
class PromptStats:
    idx: int
    token_len: int
    total_len: int
    char_len: int
    preview: str


def _load_questions(cfg: Config) -> list[str]:
    df = gsm8k(cfg) if cfg.qa_dataset == "gsm8k" else get_qa_dataset(cfg)
    return [row.question for row in df.itertuples()]  # type: ignore[attr-defined]


def _prepare_prompt(tokenizer, model_path: str, prompt: str) -> str:
    if "instruct" in model_path.lower():
        message = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    return prompt


def _summarize(cfg: Config, stats: list[PromptStats]) -> None:
    if not stats:
        print("No questions found.")
        return

    token_lengths = [item.token_len for item in stats]
    total_lengths = [item.total_len for item in stats]
    char_lengths = [item.char_len for item in stats]

    print("\nSummary")
    print("=" * 80)
    print(f"questions checked : {len(stats)}")
    print(f"qa_dataset        : {cfg.qa_dataset}")
    print(f"qa_n_shots        : {cfg.qa_n_shots}")
    print(f"gen_length        : {cfg.gen_length}")
    print(f"max prompt tokens : {max(token_lengths)}")
    print(f"max total tokens  : {max(total_lengths)}")
    print(f"max prompt chars  : {max(char_lengths)}")

    top = sorted(stats, key=lambda item: item.token_len, reverse=True)[:10]
    print("\nTop 10 longest prompts by token length")
    print("=" * 80)
    for item in top:
        print(
            f"{item.idx:>4} | prompt_tokens={item.token_len:>5} | total={item.total_len:>5} "
            f"| chars={item.char_len:>5} | {item.preview}",
        )


def main() -> None:
    cfg = Config()
    assert cfg.model == "llada", "Use this script with model=llada"

    tokenizer = get_tokenizer(cfg, "llada")
    questions = _load_questions(cfg)

    if cfg.qa_dataset_len > 0:
        limit = min(cfg.qa_dataset_len, DEFAULT_LIMIT, len(questions))
    else:
        limit = min(DEFAULT_LIMIT, len(questions))

    stats: list[PromptStats] = []

    print("Per-question prompt lengths")
    print("=" * 80)
    print(" idx | prompt_tokens | total_with_gen | chars | preview")
    print("-" * 80)

    for idx, prompt in enumerate(questions[:limit]):
        prompt_str = _prepare_prompt(tokenizer, cfg.llada_model_path, prompt)
        encoded = tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        token_len = int(encoded["input_ids"].shape[1])
        total_len = token_len + cfg.gen_length
        char_len = len(prompt_str)
        preview = " ".join(prompt_str.split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[: PREVIEW_CHARS - 3] + "..."

        item = PromptStats(
            idx=idx,
            token_len=token_len,
            total_len=total_len,
            char_len=char_len,
            preview=preview,
        )
        stats.append(item)
        print(
            f"{item.idx:>4} | {item.token_len:>13} | {item.total_len:>14} | {item.char_len:>5} | {item.preview}",
        )

    _summarize(cfg, stats)


if __name__ == "__main__":
    main()

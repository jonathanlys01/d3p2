import torch

from d5p4.code_eval import CodeEvaluator
from d5p4.dream_code import decode_bodies, humaneval_instruct_body, humaneval_instruct_prompt


PROBLEM = 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'
TESTS = ["def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"]


def test_humaneval_instruct_turns_match_lm_eval():
    user, prefix = humaneval_instruct_prompt(PROBLEM)

    assert user == f"Write a solution to the following problem and make sure that it passes the tests:\n```{PROBLEM}"
    assert prefix == f"Here is the completed function:\n```python\n{PROBLEM}\n"


def test_body_is_cut_at_the_fence_and_the_stop_strings():
    assert humaneval_instruct_body("    return a + b\n```\n\nThis adds the numbers.") == "    return a + b\n"
    assert humaneval_instruct_body("    return a + b\n\n\nprint(add(1, 2))") == "    return a + b\n\n"
    assert humaneval_instruct_body("    return a + b\n\ndef helper():\n    pass") == "    return a + b\n"


class _CharTokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|im_end|>" else 0

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


def test_decoded_bodies_keep_their_indentation_and_pass_the_tests():
    body = "    return a + b\n```"
    raw = torch.tensor([[5, 6, *[ord(c) for c in body], 1, ord("x")]])

    bodies = decode_bodies(_CharTokenizer(), raw, prompt_len=2)

    assert bodies == ["    return a + b\n"]
    result = CodeEvaluator(timeout_s=10.0).validate(
        bodies[0], prompt=PROBLEM, tests=TESTS, entry_point="add", dataset="humaneval"
    )
    assert result.passed

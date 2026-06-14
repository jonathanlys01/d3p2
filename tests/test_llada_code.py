import os
import sys
import tempfile
import unittest
import uuid

import torch


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.llada_code import _decode_generations, save


class _FakeTokenizer:
    def decode(self, tokens: list[int], skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


class _FakeSampler:
    tokenizer = _FakeTokenizer()

    def _preprocess_prompt(self, prompt: str):
        del prompt
        return torch.tensor([[1, 2]], dtype=torch.long)


class TestLladaCodeRunner(unittest.TestCase):
    def test_decode_generations_strips_prompt_tokens(self):
        raw_samples = torch.tensor(
            [
                [1, 2, ord("o"), ord("k")],
                [1, 2, ord("n"), ord("o")],
            ],
            dtype=torch.long,
        )

        decoded = _decode_generations(_FakeSampler(), "prompt", raw_samples)

        self.assertEqual(decoded, ["ok", "no"])

    def test_save_writes_canonical_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(disable_sys_args=True, results_dir=tmpdir)
            results = [
                {
                    "task_id": "1",
                    "prompt": "Write add.",
                    "reference_code": "def add(a, b): return a + b",
                    "tests": ["assert add(1, 2) == 3"],
                    "entry_point": "",
                    "dataset": "mbpp",
                    "generations": ["def add(a, b): return a + b"],
                    "validation": [],
                    "scores": [1],
                    "accuracy": 1.0,
                },
            ]

            save(results, cfg, uuid.uuid4(), internal_scores=[[0.5]])

            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("temp_code_"))


if __name__ == "__main__":
    unittest.main()

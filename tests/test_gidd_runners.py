import os
import sys
import tempfile
import types
import uuid

import torch

from d5p4.config import Config
from d5p4.gidd_code import _decode_generations as decode_code_generations
from d5p4.gidd_code import save as save_code
from d5p4.gidd_math import _decode_generations as decode_math_generations
from d5p4.gidd_math import save as save_math


class _FakeTokenizer:
    def decode(self, tokens: list[int], skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


class _FakeSampler:
    tokenizer = _FakeTokenizer()

    def _preprocess_prompt(self, prompt: str):
        del prompt
        return torch.tensor([[1, 2]], dtype=torch.long)


def test_gidd_math_decode_generations_strips_prompt_tokens():
    raw_samples = torch.tensor(
        [
            [1, 2, ord("4"), ord("2")],
            [1, 2, ord("1"), ord("7")],
        ],
        dtype=torch.long,
    )

    decoded = decode_math_generations(_FakeSampler(), "prompt", raw_samples)

    assert decoded == ["42", "17"]


def test_gidd_code_decode_generations_strips_prompt_tokens():
    raw_samples = torch.tensor(
        [
            [1, 2, ord("o"), ord("k")],
            [1, 2, ord("n"), ord("o")],
        ],
        dtype=torch.long,
    )

    decoded = decode_code_generations(_FakeSampler(), "prompt", raw_samples)

    assert decoded == ["ok", "no"]


def test_gidd_math_save_writes_canonical_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(disable_sys_args=True, model="gidd", results_dir=tmpdir)
        results = [
            {
                "question": "What is 40 + 2?",
                "gold_answer": "42",
                "answer_str": "42",
                "generations": ["42"],
                "scores": [1],
                "accuracy": 1.0,
            },
        ]

        save_math(results, cfg, uuid.uuid4())

        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith("temp_math_")


def test_gidd_code_save_writes_canonical_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(disable_sys_args=True, model="gidd", results_dir=tmpdir)
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

        save_code(results, cfg, uuid.uuid4())

        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith("temp_gidd_code_")


def test_single_run_gidd_passes_config_prompt(monkeypatch):
    seen: dict[str, object] = {}

    class _SingleRunTokenizer:
        def batch_decode(self, samples, skip_special_tokens=True):
            del skip_special_tokens
            return [str(row.tolist()) for row in samples]

    class _SingleRunSampler:
        distributed_utils = None
        tokenizer = _SingleRunTokenizer()

        def __init__(self, config):
            seen["config"] = config
            self.model = object()

        def sample(self, prompt=None):
            seen["prompt"] = prompt
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        prompt="Check conditioned generation.",
        n_runs=1,
        skip_eval=True,
        results_dir=tempfile.mkdtemp(),
    )

    fake_common_exps = types.ModuleType("d5p4.common_exps")
    fake_common_exps.eval_samples = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "d5p4.common_exps", fake_common_exps)
    from d5p4 import single_run_gidd

    monkeypatch.setattr(single_run_gidd, "Config", lambda: cfg)
    monkeypatch.setattr(single_run_gidd, "GIDDSampler", _SingleRunSampler)
    monkeypatch.setattr(single_run_gidd, "compile_model", lambda model, _config: model)
    monkeypatch.setattr(single_run_gidd, "seed_all", lambda _seed: None)
    monkeypatch.setattr(single_run_gidd, "build_generation_result_payload", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(single_run_gidd, "print", lambda *_args, **_kwargs: None)

    single_run_gidd.main()

    assert seen["prompt"] == cfg.prompt

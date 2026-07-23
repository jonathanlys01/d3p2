import os
import tempfile
import types
import uuid
from typing import cast

import torch

from d5p4.config import Config
from d5p4.data.math_ds import _format_dream_gsm8k_query
from d5p4.diffusion_dream import DreamSampler
from d5p4.dream_math import save as save_math
from d5p4.single_run_dream import _decode_generations


class _Tokenizer:
    eos_token_id = 8
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 7 if token == "<|im_end|>" else 0

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids if token_id not in {7, 8})


class _FakeSampler:
    tokenizer = _Tokenizer()

    def _preprocess_prompt(self, prompt):
        assert prompt == "question"
        return torch.tensor([[1, 2]])


def test_dream_decode_strips_prompt_and_stops_at_end_tokens():
    raw_samples = torch.tensor(
        [
            [1, 2, ord("4"), ord("2"), 8, ord("x")],
            [1, 2, ord("1"), ord("7"), 7, ord("y")],
        ],
    )

    decoded = _decode_generations(cast(DreamSampler, _FakeSampler()), "question", raw_samples)

    assert decoded == ["42", "17"]


def test_dream_resume_decode_uses_stored_prompt_length():
    class _ResumeSampler(_FakeSampler):
        def _preprocess_prompt(self, prompt):
            raise AssertionError("resumed generations must reuse the stored prompt length")

    raw_samples = torch.tensor([[99, 1, 2, ord("4"), ord("2"), 8]])

    decoded = _decode_generations(
        cast(DreamSampler, _ResumeSampler()),
        "question",
        raw_samples,
        prompt_len=3,
    )

    assert decoded == ["42"]


def test_dream_decode_keeps_content_after_leading_stop_markers():
    raw_samples = torch.tensor([[1, 2, 7, 8, ord("4"), ord("2"), 8]])

    decoded = _decode_generations(cast(DreamSampler, _FakeSampler()), "question", raw_samples)

    assert decoded == ["42"]


def test_dream_gsm8k_prompt_matches_official_zero_shot_profile():
    assert _format_dream_gsm8k_query("What is 40 + 2?") == "Q: What is 40 + 2?\n\nA:"


def test_dream_math_save_writes_canonical_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(disable_sys_args=True, model="dream", results_dir=tmpdir)
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

        save_math(results, config, uuid.uuid4(), internal_scores=[[0.9]])

        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith("temp_dream_math_")


def test_single_run_dream_routes_direct_prompt(monkeypatch):
    from d5p4 import single_run_dream

    seen = {}

    class _Sampler:
        distributed_utils = None
        tokenizer = _Tokenizer()

        def __init__(self, config):
            self.model = object()
            self.config = config

        def _preprocess_prompt(self, prompt):
            seen["preprocessed"] = prompt
            return torch.tensor([[1, 2]])

        def sample(self, prompt, return_internal_scores=False):
            seen["prompt"] = prompt
            assert return_internal_scores
            return torch.tensor([[1, 2, ord("o"), ord("k")]]), torch.tensor([0.5])

    config = Config(
        disable_sys_args=True,
        model="dream",
        prompt="Direct prompt",
        gen_length=2,
        n_groups=1,
        group_size=1,
        skip_eval=True,
        resume_runs=False,
        results_dir=tempfile.mkdtemp(),
        standalone_job=True,
    )

    monkeypatch.setattr(single_run_dream, "Config", lambda: config)
    monkeypatch.setattr(single_run_dream, "DreamSampler", _Sampler)
    monkeypatch.setattr(single_run_dream, "compile_model", lambda model, _config, dynamic=False: model)
    monkeypatch.setattr(single_run_dream, "seed_all", lambda _seed: None)
    monkeypatch.setattr(single_run_dream, "print", lambda *_args, **_kwargs: None)

    single_run_dream.main()

    assert seen["prompt"] == "Direct prompt"

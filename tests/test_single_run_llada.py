import os
import sys
import unittest
from types import MethodType, SimpleNamespace

import torch
from torch import nn


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler
from d5p4.single_run_llada import _select_group_representatives


MASK_INDEX = 99
PROMPT_TOKENS = torch.tensor([[7]], dtype=torch.long)
VOCAB_SIZE = 3


def _build_test_sampler(config: Config) -> LLADASampler:
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.mask_index = MASK_INDEX
    sampler.sequence_length = PROMPT_TOKENS.shape[1] + config.gen_length
    sampler.selector = SimpleNamespace(distributed_utils=None)
    sampler.distributed_utils = None
    sampler.forward_calls = 0

    def _preprocess_prompt(_self, prompt: str) -> torch.Tensor:
        del prompt
        return PROMPT_TOKENS.clone()

    def _forward_model(self, x: torch.Tensor, *, output_hidden_states: bool = True, logits_slice: slice | None = None):
        self.forward_calls += 1
        logits = torch.zeros((x.shape[0], x.shape[1], VOCAB_SIZE), dtype=torch.float32)
        logits[:, :, 0] = 3.0
        logits[:, :, 1] = 1.0
        if logits_slice is not None:
            logits = logits[:, logits_slice]
        embeddings = [torch.zeros((x.shape[0], x.shape[1], 1), dtype=torch.float32)] if output_hidden_states else None
        return logits, embeddings

    sampler._preprocess_prompt = MethodType(_preprocess_prompt, sampler)
    sampler._forward_model = MethodType(_forward_model, sampler)
    return sampler


class TestSingleRunLlada(unittest.TestCase):
    def test_select_group_representatives_uses_best_score_per_group(self):
        texts = ["a0", "a1", "b0", "b1", "c0", "c1"]
        scores = [0.2, 0.8, 0.7, 0.3, 0.1, 0.9]

        selected, selected_indices = _select_group_representatives(texts, scores, group_size=2)

        self.assertEqual(selected, ["a1", "b0", "c1"])
        self.assertEqual(selected_indices, [1, 2, 5])

    def test_select_group_representatives_requires_aligned_scores(self):
        with self.assertRaisesRegex(ValueError, "one score per generated sequence"):
            _select_group_representatives(["a0", "a1"], [0.1], group_size=2)

    def test_select_group_representatives_requires_complete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            _select_group_representatives(["a0", "a1", "b0"], [0.1, 0.2, 0.3], group_size=2)

    def test_sample_returns_internal_scores_without_extra_forward(self):
        cfg = Config(
            disable_sys_args=True,
            model="llada",
            cfg_scale=1.0,
            llada_steps=2,
            gen_length=2,
            block_length=2,
            remasking="low_confidence",
            cat_temperature=0.0,
            confidence_eos_eot_inf=False,
            logits_eos_inf=False,
            n_groups=1,
            group_size=2,
            subsample_start=10,
            subsample_end=10,
        )
        sampler = _build_test_sampler(cfg)

        samples, scores = sampler.sample("prompt", return_internal_scores=True)

        self.assertEqual(samples.shape, (2, PROMPT_TOKENS.shape[1] + cfg.gen_length))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(sampler.forward_calls, cfg.llada_steps)


if __name__ == "__main__":
    unittest.main()

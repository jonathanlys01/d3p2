import os
import sys
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.single_run_llada import _final_step_uses_cfg, _select_group_representatives


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

    def test_final_step_uses_cfg_handles_default_guidance_end(self):
        cfg = Config(
            disable_sys_args=True,
            model="llada",
            cfg_scale=0.0,
            llada_steps=128,
            gen_length=128,
            block_length=32,
            guidance_start=0,
            guidance_end=-1,
        )

        self.assertTrue(_final_step_uses_cfg(cfg))


if __name__ == "__main__":
    unittest.main()

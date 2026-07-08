import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn

import d5p4.diffusion_llada as diffusion_llada_module
import d5p4.diffusion_llada_profile as diffusion_llada_profile_module
from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler
from d5p4.diffusion_llada_profile import LLADAProfilerSampler
from d5p4.llada_ref._generate import generate as ref_generate


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT / "tests" / "artifacts"
FIGURE_PATH = ARTIFACT_DIR / "llada_first_demask.svg"
MASK_INDEX = 99
PROMPT_TOKENS = torch.tensor([[7]], dtype=torch.long)
VOCAB_SIZE = 2


diffusion_llada_module.tqdm = lambda iterable, **kwargs: iterable
diffusion_llada_profile_module.tqdm = lambda iterable, **kwargs: iterable


def _make_config(remasking: str, *, cat_temperature: float = 0.0, selection_temperature: float = 1.0) -> Config:
    return Config(
        disable_sys_args=True,
        model="llada",
        llada_steps=4,
        gen_length=4,
        block_length=4,
        remasking=remasking,
        selection_temperature=selection_temperature,
        cat_temperature=cat_temperature,
        cfg_scale=1.0,
        confidence_eos_eot_inf=False,
        logits_eos_inf=False,
        n_groups=1,
        group_size=1,
        subsample_start=10,
        subsample_end=10,
    )


def _build_logits() -> torch.Tensor:
    logits = torch.zeros((1, PROMPT_TOKENS.shape[1] + 4, VOCAB_SIZE), dtype=torch.float32)
    logits[0, 1:, 0] = torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    return logits


def _build_test_sampler(sampler_cls: type[nn.Module], config: Config, logits: torch.Tensor):
    sampler = sampler_cls.__new__(sampler_cls)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.mask_index = MASK_INDEX
    sampler.sequence_length = PROMPT_TOKENS.shape[1] + config.gen_length
    sampler.selector = SimpleNamespace(distributed_utils=None)
    sampler.distributed_utils = None
    sampler.enable_profiling_scopes = False
    sampler.cuda_timer = diffusion_llada_profile_module.CUDAScopeTimer()
    sampler.first_demasked_position = None

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        return PROMPT_TOKENS.clone()

    def _forward_model(self, x: torch.Tensor, *, output_hidden_states: bool = True, logits_slice: slice | None = None):
        if self.first_demasked_position is None:
            gen_tokens = x[0, PROMPT_TOKENS.shape[1] :]
            demasked = torch.nonzero(gen_tokens != self.mask_index, as_tuple=False).squeeze(-1)
            if demasked.numel() == 1:
                self.first_demasked_position = int(demasked.item())
        out_logits = logits.clone() if logits_slice is None else logits[:, logits_slice].clone()
        embeddings = [torch.zeros((x.shape[0], x.shape[1], 1), dtype=torch.float32)] if output_hidden_states else None
        return out_logits, embeddings

    sampler._preprocess_prompt = MethodType(_preprocess_prompt, sampler)
    sampler._forward_model = MethodType(_forward_model, sampler)
    return sampler


class _FakeRefModel:
    def __init__(self, logits: torch.Tensor):
        self.device = "cpu"
        self.logits = logits
        self.first_demasked_position = None

    def __call__(self, x: torch.Tensor, attention_mask=None):
        if self.first_demasked_position is None:
            gen_tokens = x[0, PROMPT_TOKENS.shape[1] :]
            demasked = torch.nonzero(gen_tokens != MASK_INDEX, as_tuple=False).squeeze(-1)
            if demasked.numel() == 1:
                self.first_demasked_position = int(demasked.item())
        return SimpleNamespace(logits=self.logits.clone())


def _run_main_or_profile(
    sampler_cls: type[nn.Module],
    remasking: str,
    *,
    seed: int,
    selection_temperature: float = 0.25,
) -> int:
    config = _make_config(
        remasking,
        cat_temperature=0.0,
        selection_temperature=selection_temperature,
    )
    sampler = _build_test_sampler(sampler_cls, config, _build_logits())
    torch.manual_seed(seed)
    out = sampler.sample("prompt")
    assert out.shape == (1, PROMPT_TOKENS.shape[1] + config.gen_length)
    assert sampler.first_demasked_position is not None
    return sampler.first_demasked_position


def _run_reference(remasking: str, *, seed: int) -> int:
    model = _FakeRefModel(_build_logits())
    torch.manual_seed(seed)
    out = ref_generate(
        model=model,
        prompt=PROMPT_TOKENS.clone(),
        attention_mask=None,
        steps=4,
        gen_length=4,
        block_length=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking=remasking,
        mask_id=MASK_INDEX,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
    )
    assert out.shape == (1, PROMPT_TOKENS.shape[1] + 4)
    assert model.first_demasked_position is not None
    return model.first_demasked_position


def _counts_for(mode: str, impl: str, seeds: range) -> list[int]:
    counts = [0, 0, 0, 0]
    for seed in seeds:
        if impl == "main":
            pos = _run_main_or_profile(LLADASampler, mode, seed=seed)
        elif impl == "profile":
            pos = _run_main_or_profile(LLADAProfilerSampler, mode, seed=seed)
        elif impl == "reference":
            pos = _run_reference(mode, seed=seed)
        else:
            raise ValueError(impl)
        counts[pos] += 1
    return counts


def _write_svg(counts_by_mode: dict[str, list[int]], output_path: Path) -> None:
    width = 840
    margin_left = 110
    margin_top = 660
    bar_height = 24
    row_gap = 52
    max_bar_width = 480
    scale = max(max(counts) for counts in counts_by_mode.values())
    content_height = margin_top + len(counts_by_mode) * (4 * bar_height + row_gap) + 24
    height = max(220, content_height)
    logits = _build_logits()[0, 1:, 0]
    confidence = torch.softmax(_build_logits()[0, 1:, :], dim=-1)[:, 0]
    top2_confidence = confidence[:2]
    selection_probs = torch.softmax(top2_confidence / 0.25, dim=0)
    logits_text = ", ".join(f"{value:.1f}" for value in logits.tolist())
    confidence_text = ", ".join(f"{value:.3f}" for value in confidence.tolist())
    selection_text = ", ".join(f"{value:.3f}" for value in selection_probs.tolist())
    logits_min = float(logits.min().item())
    logits_max = float(logits.max().item())

    def _normalize(values: torch.Tensor) -> list[float]:
        if values.numel() == 0:
            return []
        min_val = float(values.min().item())
        max_val = float(values.max().item())
        if max_val == min_val:
            return [1.0 for _ in values]
        return [float((value.item() - min_val) / (max_val - min_val)) for value in values]

    logits_norm = _normalize(logits)
    confidence_norm = [float(value.item()) for value in confidence]
    selection_norm = [float(value.item()) for value in selection_probs]

    viz_left = 40
    slot_width = 124
    viz_bar_width = 68
    viz_max_height = 100
    value_gap = 28
    base_y = {"logits": 110, "confidence": 300, "selection": 490}

    viz_rows = [
        ("raw logits", logits_norm, logits.tolist(), base_y["logits"], "#2563eb"),
        ("confidence p(x0)", confidence_norm, confidence.tolist(), base_y["confidence"], "#059669"),
        ("top-2 selection p", selection_norm, selection_probs.tolist(), base_y["selection"], "#d97706"),
    ]

    rows = [
        "<style>",
        '  text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }',
        "  .title { font-size: 15px; font-weight: 700; fill: #111827; }",
        "  .subtitle { font-size: 11px; fill: #4b5563; }",
        "  .label { font-size: 12px; font-weight: 600; fill: #374151; }",
        "  .axis-label { font-size: 11px; fill: #6b7280; }",
        "  .value-label { font-size: 11px; fill: #111827; font-weight: 600; }",
        "  .meta-info { font-size: 10px; fill: #9ca3af; }",
        "  .category-title { font-size: 13px; font-weight: 700; fill: #111827; }",
        "</style>",
    ]
    rows.append('<text x="12" y="24" class="title">First demasked position for logits [4.0, 2.0, 1.0, 0.0]</text>')
    rows.append(
        f'<text x="12" y="42" class="subtitle">raw logits: [{logits_text}]   confidence: [{confidence_text}]</text>',
    )
    rows.append(
        f'<text x="12" y="58" class="subtitle">selection probs over top-2 @ T=0.25: [{selection_text}]</text>',
    )

    for label, normalized_values, raw_values, y, color in viz_rows:
        rows.append(
            f'<text x="12" y="{y - 12}" class="label">{label}</text>',
        )
        for idx, (norm_value, raw_value) in enumerate(zip(normalized_values, raw_values, strict=False)):
            x = viz_left + 140 + idx * slot_width
            bar_h = max(2, int(viz_max_height * norm_value))
            top = y + (viz_max_height - bar_h)
            rows.append(
                f'<rect x="{x}" y="{top}" width="{viz_bar_width}" height="{bar_h}" fill="{color}" rx="4" fill-opacity="0.9" />',
            )
            rows.append(
                f'<text x="{x + viz_bar_width / 2}" y="{y + viz_max_height + 20}" class="axis-label" text-anchor="middle">p{idx}</text>',
            )
            rows.append(
                f'<text x="{x + viz_bar_width / 2}" y="{y + viz_max_height + 20 + value_gap}" class="value-label" text-anchor="middle">{raw_value:.3f}</text>',
            )

    rows.append(
        f'<text x="660" y="{base_y["logits"] + 30}" class="meta-info">min-max norm [{logits_min:.1f}, {logits_max:.1f}]</text>',
    )
    rows.append(
        f'<text x="660" y="{base_y["confidence"] + 30}" class="meta-info">already normalized [0, 1]</text>',
    )
    rows.append(
        f'<text x="660" y="{base_y["selection"] + 30}" class="meta-info">restricted to top-2</text>',
    )

    rows.append(
        f'<line x1="12" y1="{margin_top - 12}" x2="{width - 12}" y2="{margin_top - 12}" stroke="#e5e7eb" stroke-width="1.5" />',
    )

    for row_idx, (mode, counts) in enumerate(counts_by_mode.items()):
        y0 = margin_top + row_idx * (4 * bar_height + row_gap)
        rows.append(
            f'<text x="12" y="{y0 + 12}" class="category-title">{mode}</text>',
        )
        for pos_idx, count in enumerate(counts):
            y = y0 + 24 + pos_idx * bar_height
            width_px = 0 if scale == 0 else int(max_bar_width * count / scale)
            fill = "#059669" if count else "#f3f4f6"
            rows.append(
                f'<text x="{margin_left - 48}" y="{y + 14}" class="axis-label">p{pos_idx}</text>',
            )
            rows.append(
                f'<rect x="{margin_left}" y="{y}" width="{width_px}" height="16" fill="{fill}" rx="3" />',
            )
            if count > 0:
                rows.append(
                    f'<text x="{margin_left + width_px + 10}" y="{y + 13}" class="value-label">{count}</text>',
                )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            *rows,
            "</svg>",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


class TestLLADASelectionTemperature(unittest.TestCase):
    def test_selection_temperature_confidence_matches_low_confidence(self):
        logits = _build_logits()
        x0 = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        prompt_len = PROMPT_TOKENS.shape[1]

        low_conf_config = _make_config("low_confidence")
        sel_temp_config = _make_config("selection_temperature", selection_temperature=0.25)

        for sampler_cls in (LLADASampler, LLADAProfilerSampler):
            low_conf_sampler = _build_test_sampler(sampler_cls, low_conf_config, logits)
            sel_temp_sampler = _build_test_sampler(sampler_cls, sel_temp_config, logits)

            low_conf = low_conf_sampler._get_confidence(logits.clone(), x0, num_block=0, prompt_len=prompt_len)
            sel_temp = sel_temp_sampler._get_confidence(logits.clone(), x0, num_block=0, prompt_len=prompt_len)

            self.assertTrue(torch.equal(low_conf, sel_temp))
            self.assertTrue(torch.isfinite(low_conf[:, 1:]).all())

    def test_low_confidence_and_random_match_reference_across_all_implementations(self):
        seeds = range(16)
        for mode in ("low_confidence", "random"):
            main_positions = [_run_main_or_profile(LLADASampler, mode, seed=seed) for seed in seeds]
            profile_positions = [_run_main_or_profile(LLADAProfilerSampler, mode, seed=seed) for seed in seeds]
            reference_positions = [_run_reference(mode, seed=seed) for seed in seeds]

            self.assertEqual(main_positions, profile_positions)
            self.assertEqual(main_positions, reference_positions)

    def test_selection_temperature_matches_profile_and_only_samples_from_top_2k(self):
        seeds = range(64)
        main_positions = [
            _run_main_or_profile(LLADASampler, "selection_temperature", seed=seed, selection_temperature=0.25)
            for seed in seeds
        ]
        profile_positions = [
            _run_main_or_profile(LLADAProfilerSampler, "selection_temperature", seed=seed, selection_temperature=0.25)
            for seed in seeds
        ]

        self.assertEqual(main_positions, profile_positions)
        self.assertTrue(set(main_positions).issubset({0, 1}))
        self.assertIn(0, main_positions)
        self.assertIn(1, main_positions)

    def test_generates_first_demask_figure(self):
        seeds = range(64)
        counts_by_mode = {
            "low_confidence": _counts_for("low_confidence", "main", seeds),
            "random": _counts_for("random", "main", seeds),
            "selection_temperature": _counts_for("selection_temperature", "main", seeds),
        }
        _write_svg(counts_by_mode, FIGURE_PATH)

        self.assertEqual(counts_by_mode["low_confidence"], [64, 0, 0, 0])
        self.assertEqual(sum(counts_by_mode["random"]), 64)
        self.assertEqual(sum(counts_by_mode["selection_temperature"]), 64)
        self.assertEqual(counts_by_mode["selection_temperature"][2:], [0, 0])
        self.assertTrue(FIGURE_PATH.exists())
        figure_text = FIGURE_PATH.read_text(encoding="utf-8")
        self.assertIn("<svg", figure_text)
        self.assertIn("raw logits", figure_text)
        self.assertIn("confidence p(x0)", figure_text)
        self.assertIn('fill="#2563eb"', figure_text)


if __name__ == "__main__":
    unittest.main()

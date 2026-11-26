"""Opt-in dump of every LLaDA selection step, for offline solver and quality-proxy analyses (revision item A3).

With `selection_dump=true`, `LLADASampler.sample` writes one compressed `.npz` per question to
`<results_dir>/selection_dumps/<sha1(prompt)[:16]>.npz`. Nothing here runs, and nothing is allocated, when the flag is
off. Recording draws no random numbers and does not touch the sampler state, so a dumped run follows the same
trajectory as an undumped one with the same seed.

Arrays (T = selection-window steps of the question, B = candidate pool k * w, k = selected rows per step):

- `sel` int16 [T, k]: rows the production selector returned at step t, in solver order; -1 rows outside the window.
- `selected` bool [T]: whether step t ran the selector (outside `subsample_start..subsample_end` it does not).
- `mask_frac` float32 [T]: fraction of the generation still masked in the candidates scored at step t.
- `block_mask_frac` float32 [T]: same, within the current block (what the selector sees).
- `score_neg_entropy` float32 [T, B]: raw quality score before any normalization: minus the mean per-position
  entropy over the selector window (the paper's Q is a min-max normalization of this, per step).
- `score_confidence` float32 [T, B]: mean max-probability over the selector window (an alternative raw proxy).
- `kernel_steps` int32 [M]: steps whose kernels are stored (every `selection_dump_every`-th selection step).
- `L` float32 [M, B, B]: the kernel the selector built at that step (`selector.compute_kernel`), exactly what the
  production solver maximizes over transversals.
- `S` float32 [M, B, B]: cosine similarity of the flattened, normalized selector embeddings.
- `final_tokens` int32 [B, gen_length], `final_internal_scores` float64 [B]: the run's outputs, in result order;
  the internal scores join a dump to its question in the generation JSON.
- `prompt` str, `meta` str (JSON): configuration needed to read the arrays.

Ancestry: candidate row r at step t + 1 descends from row `sel[t, r // w]` at step t (identity on non-selected steps).
Kernels are float32: neighbouring candidates have cosines of 0.99 and above, which float16 would round away.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from d5p4.config import Cache, Config


DUMP_SUBDIR = "selection_dumps"
FORMAT_VERSION = 1


def dump_path(config: Config, prompt: str) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return Path(config.results_dir) / DUMP_SUBDIR / f"{digest}.npz"


def raw_scores(log_p_x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-candidate minus mean entropy and mean max-probability over the window, before any normalization."""
    log_p = log_p_x0.float()
    p = log_p.exp()
    neg_entropy = (p * log_p).nan_to_num(0.0).sum(dim=-1).mean(dim=-1)
    confidence = p.max(dim=-1).values.mean(dim=-1)
    return neg_entropy, confidence


def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    flat = F.normalize(embeddings.float().reshape(embeddings.size(0), -1), dim=-1, eps=1e-12)
    return flat @ flat.T


class SelectionDumpRecorder:
    """Collects one question's selection steps on the host and writes them once the question is done."""

    def __init__(self, config: Config, prompt: str, prompt_len: int, selector: Any):
        self.config = config
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.selector = selector
        self.every = config.selection_dump_every
        self.sel: list[np.ndarray] = []
        self.selected: list[bool] = []
        self.mask_frac: list[float] = []
        self.block_mask_frac: list[float] = []
        self.neg_entropy: list[np.ndarray] = []
        self.confidence: list[np.ndarray] = []
        self.kernel_steps: list[int] = []
        self.kernels: list[np.ndarray] = []
        self.similarities: list[np.ndarray] = []
        self._selection_count = 0

    @classmethod
    def create(cls, config: Config, prompt: str, prompt_len: int, selector: Any, distributed: bool):
        """A recorder when `selection_dump` is on, else None (the sampler then skips every recording call)."""
        if not config.selection_dump:
            return None
        if distributed:
            raise RuntimeError("selection_dump is single-process only; set standalone_job=true")
        return cls(config, prompt, prompt_len, selector)

    @torch.no_grad()
    def record(self, cache: Cache, slice_idx: torch.Tensor, subsample_step: bool, x: torch.Tensor, mask_id: int):
        """Record one step: `x` is the full candidate state before this step's expansion and unmasking."""
        assert cache.log_p_x0 is not None and cache.x is not None
        n_select = self.config.n_groups
        sel = np.full(n_select, -1, dtype=np.int16)
        if subsample_step:
            sel[:] = slice_idx.detach().cpu().numpy().astype(np.int16)
        self.sel.append(sel)
        self.selected.append(bool(subsample_step))

        generation = x[0, self.prompt_len :]
        self.mask_frac.append(float((generation == mask_id).float().mean()))
        self.block_mask_frac.append(float((cache.x[0] == mask_id).float().mean()))

        neg_entropy, confidence = raw_scores(cache.log_p_x0)
        self.neg_entropy.append(neg_entropy.cpu().numpy().astype(np.float32))
        self.confidence.append(confidence.cpu().numpy().astype(np.float32))

        step = len(self.sel) - 1
        if subsample_step and cache.embeddings is not None and hasattr(self.selector, "compute_kernel"):
            if self._selection_count % self.every == 0:
                kernel = self.selector.compute_kernel(cache)
                if kernel is not None:
                    self.kernel_steps.append(step)
                    self.kernels.append(kernel.float().cpu().numpy())
                    self.similarities.append(cosine_similarity(cache.embeddings).cpu().numpy())
            self._selection_count += 1

    def meta(self) -> dict[str, Any]:
        c = self.config
        return {
            "format_version": FORMAT_VERSION,
            "method": c.method,
            "transversal": c.transversal,
            "n_groups": c.n_groups,
            "group_size": c.group_size,
            "w_interaction": c._w_interaction,
            "kernel_method": c._kernel_method,
            "kernel_type": c._kernel_type,
            "score_method": c._score_method,
            "cat_temperature": c.cat_temperature,
            "remasking": c.remasking,
            "seed": c.seed,
            "gen_length": c.gen_length,
            "block_length": c.block_length,
            "llada_steps": c.llada_steps,
            "qa_dataset": c.qa_dataset,
            "selection_dump_every": self.every,
            "prompt_len": self.prompt_len,
        }

    def save(self, x: torch.Tensor, final_internal_scores: torch.Tensor | None) -> Path:
        path = dump_path(self.config, self.prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_pool = self.config.batch_size
        arrays: dict[str, Any] = {
            "sel": np.stack(self.sel) if self.sel else np.zeros((0, self.config.n_groups), np.int16),
            "selected": np.asarray(self.selected, dtype=bool),
            "mask_frac": np.asarray(self.mask_frac, dtype=np.float32),
            "block_mask_frac": np.asarray(self.block_mask_frac, dtype=np.float32),
            "score_neg_entropy": np.stack(self.neg_entropy) if self.neg_entropy else np.zeros((0, n_pool), np.float32),
            "score_confidence": np.stack(self.confidence) if self.confidence else np.zeros((0, n_pool), np.float32),
            "kernel_steps": np.asarray(self.kernel_steps, dtype=np.int32),
            "L": np.stack(self.kernels) if self.kernels else np.zeros((0, n_pool, n_pool), np.float32),
            "S": np.stack(self.similarities) if self.similarities else np.zeros((0, n_pool, n_pool), np.float32),
            "final_tokens": x[:, self.prompt_len :].detach().cpu().numpy().astype(np.int32),
            "final_internal_scores": (
                np.asarray(final_internal_scores.detach().cpu().tolist(), dtype=np.float64)
                if final_internal_scores is not None
                else np.zeros(0, np.float64)
            ),
            "prompt": np.asarray(self.prompt),
            "meta": np.asarray(json.dumps(self.meta())),
        }
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        return path


def load_dump(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    out["prompt"] = str(out["prompt"])
    out["meta"] = json.loads(str(out["meta"]))
    return out

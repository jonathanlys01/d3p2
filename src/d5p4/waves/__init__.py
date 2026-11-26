"""Experiment waves for the D5P4 revision; see `d5p4.waves.spec` and `.jz_waves/README.md`."""

from __future__ import annotations

from collections.abc import Callable

from d5p4.waves import a2, a3, b1, b2, b4, b5, b6, e1, e2, m0, p1, parallel, s1, scale, scale_gsm8k, transfer
from d5p4.waves.spec import Run


WAVES: dict[str, Callable[[], list[Run]]] = {
    "m0": m0.runs,
    "transfer": transfer.runs,
    "p1": p1.runs,
    "s1": s1.runs,
    "e1": e1.runs,
    "e2": e2.runs,
    "b1": b1.runs,
    "b2": b2.runs,
    "b2_full": b2.full_runs,
    "scale": scale.runs,
    "a3": a3.runs,
    "a2": a2.runs,
    "parallel": parallel.runs,
    "b6_seed1": b6.seed_runs,
    "b6_sweep": b6.sweep_runs,
    "scale_gsm8k": scale_gsm8k.runs,
    "b5": b5.runs,
    "b5_dream": b5.dream_runs,
    "b4": b4.runs,
}


def get_wave(name: str) -> list[Run]:
    if name not in WAVES:
        raise SystemExit(f"Unknown wave {name!r}; available: {sorted(WAVES)}")
    return WAVES[name]()

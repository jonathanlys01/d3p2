"""Experiment waves for the D5P4 revision; see `d5p4.waves.spec` and `.jz_waves/README.md`."""

from __future__ import annotations

from collections.abc import Callable

from d5p4.waves import m0, transfer
from d5p4.waves.spec import Run


WAVES: dict[str, Callable[[], list[Run]]] = {
    "m0": m0.runs,
    "transfer": transfer.runs,
}


def get_wave(name: str) -> list[Run]:
    if name not in WAVES:
        raise SystemExit(f"Unknown wave {name!r}; available: {sorted(WAVES)}")
    return WAVES[name]()

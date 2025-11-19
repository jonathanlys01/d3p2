from __future__ import annotations  # Solves type hint evaluation timing

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from base import BaseSelector
    from beam import DiverseBeamSearch, GreedyBeamSearch
    from dpp_selector import DPP
    from exhaustive import Exhaustive
    from greedy_map import GreedyMAP
    from random_selector import RandomSelection

    from config import Config

AVAIL = {
    "dpp": ("dpp_selector", "DPP"),
    "exhaustive": ("exhaustive", "Exhaustive"),
    "greedy_map": ("greedy_map", "GreedyMAP"),
    "greedy_beam": ("beam", "GreedyBeamSearch"),
    "diverse_beam": ("beam", "DiverseBeamSearch"),
    "random": ("random_selector", "RandomSelection"),
}


def get_subsample_selector(
    config: Config,
) -> "GreedyBeamSearch | BaseSelector | DiverseBeamSearch | DPP | Exhaustive | GreedyMAP | RandomSelection":
    """
    Factory function to dynamically load and instantiate a subset selector.
    """
    name = config.method.lower()
    import_info = AVAIL.get(name)

    if import_info is None:
        raise ValueError(f"Unknown subset selector: {name}")

    module_path, class_name = import_info

    try:
        # Note: package=__name__ is only needed for relative imports (starting with .)
        # Since your AVAIL dict uses absolute paths ("subsample.dpp"), you can remove it
        # or ensure your AVAIL paths are relative (e.g., ".dpp").
        module = importlib.import_module(module_path)
        selector_class = getattr(module, class_name)

    except ImportError as e:
        raise ImportError(
            f"Could not lazily import subset selector '{name}'. "
            f"Failed to import {class_name} from {module_path}. Error: {e}",
        )
    except AttributeError:
        raise AttributeError(
            f"Could not find class '{class_name}' in module '{module_path}'.",
        )

    return selector_class(config)

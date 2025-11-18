import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from config import Config

    from .dpp import DPP
    from .greedy_map import GreedyMAP
    from .mmr import MMR
    from .monte_carlo import DiverseBeamSearch, MCSearch
    from .random import RandomSelection

AVAIL = {
    "greedy_map": (".greedy_map", "GreedyMAP"),
    "monte_carlo": (".monte_carlo", "MCSearch"),
    "diverse_beam_search": (".monte_carlo", "DiverseBeamSearch"),
    "dpp": (".dpp", "DPP"),
    "mmr": (".mmr", "MMR"),
    "random": (".random", "RandomSelection"),
}


def get_subsample_selector(
    name: str,
    config: "Config",
) -> "GreedyMAP" | "MCSearch" | "DiverseBeamSearch" | "DPP" | "MMR" | "RandomSelection":
    """
    Factory function to dynamically load and instantiate a subset selector.
    """
    name = name.lower()
    import_info = AVAIL.get(name)

    if import_info is None:
        raise ValueError(f"Unknown subset selector: {name}")

    module_path, class_name = import_info

    try:
        module = importlib.import_module(module_path, package=__name__)
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

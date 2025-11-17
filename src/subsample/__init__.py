from .base import BaseSubsetSelector
from .greedy_map import GreedyMAP


AVAIL = {
    "base": BaseSubsetSelector,
    "greedy_map": GreedyMAP,
}


def get_subsample_selector(name: str, config):
    name = name.lower()
    selector = AVAIL.get(name)
    if selector is None:
        raise ValueError(f"Unknown subset selector: {name}")
    return selector(config)

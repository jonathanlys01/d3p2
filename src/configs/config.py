"""
Config file for the project
"""

from dataclasses import dataclass
from enum import Enum

from omegaconf import OmegaConf


class EmbeddingType(Enum):
    """Embedding types."""

    external = "external"
    cached_last = "cached_last"
    cached_all = "cached_all"


@dataclass()
class Config:
    seed: int = 0

    # sampling
    num_steps: int = 1_024  # match sequence length
    cat_temperature: float = 1.0
    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer = "gpt2"

    # dpp
    k: int = 4
    expansion_factor: int = 2

    alpha: float = 0.1  # weight for the cosine similarity in the DPP kernel

    # utils
    cache_dir: str = "./.cache"

    def __post_init__(self):
        self.batch_size = self.k * self.expansion_factor


def get_config() -> Config:
    """
    Read args from CLI, merge with defaults and return a read-only Config object.
    """
    default_cfg = OmegaConf.structured(Config)
    sys_args = OmegaConf.from_cli()
    args = OmegaConf.merge(default_cfg, sys_args)
    return Config(**args)


default_config = Config()


def main():
    """
    Main function
    """
    config = get_config()
    print(config.__dict__)


if __name__ == "__main__":
    main()

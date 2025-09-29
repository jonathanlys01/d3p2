"""
Config file for the project
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import MISSING, OmegaConf


class EmbeddingType(Enum):
    """Embedding types."""

    external = "external"
    cached_last = "cached_last"
    cached_all = "cached_all"


@dataclass(frozen=True)
class Config:
    seed: int = 0

    # sampling
    num_steps: int = 1_024  # match sequence length
    batch_size: int = 1
    cat_temperature: float = 1.0
    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer = "gpt2"

    # embedding
    embedding_type: EmbeddingType = MISSING
    embedding_model: Optional[str] = None

    # dpp
    dpp: bool = False
    k = 10  # number of samples to keep
    k_dpp: bool = True  # true for k-DPP, false for DPP
    alpha: float = 0.1

    # utils
    cache_dir: str = "./.cache"

    def __post_init__(self):
        assert self.num_steps > 0, "num_steps must be positive"
        if self.embedding_type == EmbeddingType.external:
            assert self.embedding_model is not None, "embedding_model must be specified for external embeddings"


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

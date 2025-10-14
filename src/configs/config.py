"""
Config file for the project
"""

from dataclasses import dataclass

from omegaconf import OmegaConf


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
    dpp: bool = True

    subsample_start: int = 100
    subsample_end: int = 200

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
    print("Config file for the project")
    main()

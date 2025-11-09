import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf


SEQUENCE_LENGTH = 1_024
HIDDEN_SIZE = 768
RESULTS_DIR = "results"
CACHE_DIR = "./.cache"

CONFIG_FLAGS = ("--config", "-c", "config", "cfg")


def env_path_or(env_name: str, suffix: str, fallback: str) -> str:
    val = os.getenv(env_name)
    return str(Path(val) / suffix) if val else fallback


OmegaConf.register_new_resolver("env_path_or", env_path_or, use_cache=True)


@dataclass(frozen=True)
class Config:
    """
    Configuration for D3P2 experiments.
    Will be overridden by command-line arguments at initialization.
    """

    @property
    def sequence_length(self) -> int:
        return SEQUENCE_LENGTH

    @property
    def embedding_dim(self) -> int:
        return HIDDEN_SIZE

    disable_sys_args: bool = False

    seed: int = 0
    n_runs: int = 16
    compile_model: bool = False

    # sampling
    num_steps: int = SEQUENCE_LENGTH  # number of sampling steps
    cat_temperature: float = 1.0

    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer: str = "gpt2"

    # LLaDA
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base"
    llada_tokenizer: str = "GSAI-ML/LLaDA-8B-Base"

    # subset selection
    n_groups: int = 2
    group_size: int = 2
    split_groups: bool = True
    dpp: bool = True
    w_interaction: float = 0.1  # weight for diversity term in DPP, -1 for no quality term
    w_split: float = 0.0  # weight for split groups in DPP
    determinant_temperature: float = 1.0

    subsample_start: int = 300
    subsample_end: int = 400

    # eval
    ppl_model_id: str = "gpt2"
    cos_model_id: str = "jinaai/jina-embeddings-v2-base-en"

    # cache
    cache_dir: str = CACHE_DIR

    batch_size: int = 0  # to be set in __post_init__

    def __post_init__(self):
        if self.disable_sys_args:
            return

        self_args = OmegaConf.structured(self)
        sys_args = OmegaConf.from_cli()

        # Priority:
        # 1. Command-line config file (if any)
        # 2. Command-line args
        # 3. Default args

        if any(flag in sys_args for flag in CONFIG_FLAGS):
            flag = next(flag for flag in CONFIG_FLAGS if flag in sys_args)
            cfg_file = sys_args.pop(flag)  # remove the flag from sys_args (not in struct)
            cfg_args = OmegaConf.load(cfg_file)
            add_args = OmegaConf.merge(cfg_args, sys_args)
        else:
            add_args = sys_args

        args = OmegaConf.merge(self_args, add_args)
        self.__dict__.update(args)

        object.__setattr__(self, "batch_size", self.n_groups * self.group_size)

        if self.n_runs == 1:
            object.__setattr__(self, "interactive", True)

        if self.dpp is False:
            object.__setattr__(self, "w_interaction", 0.0)
            object.__setattr__(self, "w_split", 0.0)

    def __str__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self))


@dataclass
class Cache:
    x: Optional[torch.Tensor] = None
    log_p_x0: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


# TODO: move the structured json logic here


if __name__ == "__main__":
    print("Config file for the project")
    config = Config()
    print(OmegaConf.to_yaml(OmegaConf.structured(config)))

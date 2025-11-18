import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf

from subsample import AVAIL


if TYPE_CHECKING:
    import torch


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
    disable_sys_args: bool = False

    @property
    def sequence_length(self) -> int:
        return SEQUENCE_LENGTH

    embedding_dim: int = HIDDEN_SIZE  # Change when using different model

    seed: int = 0
    n_runs: int = 16
    compile_model: bool = True

    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer: str = "gpt2"

    # LLaDA
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base"
    llada_tokenizer: str = "GSAI-ML/LLaDA-8B-Base"

    # sampling
    num_steps: int = SEQUENCE_LENGTH  # number of sampling steps
    cat_temperature: float = 1.0

    # Source data
    data_path: str = "path_to.bin"
    initial_mask_ratio: float = 1.0  # ratio of tokens to mask at start of sampling (1.0 = all tokens masked)

    # Subset selection ###################################################################################
    method: str = "base"  # subset selection method
    transversal: bool = False  # use transversal sampling

    group_size: int = 2
    n_groups: int = 2

    # Subsample parameters (specific to each method)

    _kernel_type: str = "rbf"  # type of kernel to use in DPP
    _w_interaction: float = 0.0  # weight for diversity term in DPP, -1 for no quality term
    _w_split: float = 0.0  # weight for split groups in DPP
    _rbf_gamma: float = 1  # RBF kernel gamma parameter (when using RBF kernel)
    _temperature: float = 0.0  # temperature for any sampling
    _diversity_alpha: float = 0.0  # diversity coefficient for diverse beam search
    ######################################################################################################

    # windowing
    subsample_start: int = -1
    subsample_end: int = 2**31 - 1

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
        # 1. Command-line args
        # 2. Command-line provided config file (if any)
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

        assert 0 < self.initial_mask_ratio <= 1.0, "initial_mask_ratio must be in (0, 1]"

        object.__setattr__(self, "batch_size", self.n_groups * self.group_size)

        if self.n_runs == 1:
            object.__setattr__(self, "interactive", True)

        assert self.method in AVAIL, f"Method {self.method} not recognized. Available methods: {list(AVAIL.keys())}"

    def __str__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self))


@dataclass
class Cache:
    x: Optional["torch.Tensor"] = None
    log_p_x0: Optional["torch.Tensor"] = None
    embeddings: Optional["torch.Tensor"] = None


if __name__ == "__main__":
    print("Config file for the project")
    config = Config()
    print(OmegaConf.to_yaml(OmegaConf.structured(config)))

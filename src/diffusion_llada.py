"""
Minimalist diffusion sampler, adapted from LLADA codebase.
TODO: hack generate.py / get_log_likelihood.py in llada folder
TODO: implement a MDLM like sampling class
"""

from torch import nn
from transformers import AutoTokenizer

from config import Config
from dpp import SubsetSelector
from llada.configuration_llada import LLaDAConfig
from llada.modeling_llada import LLaDAModelLM


class LLADASampler(nn.Module):
    """LLADA Diffusion Model base class."""

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.model = LLaDAModelLM.from_pretrained(config.mdlm_model_path, cache_dir=config.cache_dir)
        self.selector = SubsetSelector(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.mdlm_tokenizer, cache_dir=config.cache_dir)

        self.model_config: LLaDAConfig = self.model.config

import torch
import transformers

from configs.config import Config


def seed_all(seed: int):
    """
    Set the seed for all random number generators.
    """
    transformers.set_seed(seed)


def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def get_tokenizer(config: Config, model: str):
    """
    Get the tokenizer from the config.
    """

    assert model in ["mdlm", "embedding"], f"model must be either 'mdlm' or 'embedding', got {model}"

    if model == "mdlm":
        path = config.mdlm_tokenizer

    elif model == "embedding":
        path = config.embedding_model if config.embedding_type == "external" else config.mdlm_model_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(path, cache_dir=config.cache_dir)

    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(f"Tokenizer must have a bos_token or cls_token: {tokenizer}")
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(f"Tokenizer must have a eos_token or sep_token: {tokenizer}")
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer

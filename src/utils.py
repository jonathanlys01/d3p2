import os
from builtins import print as bprint

import idr_torch
import torch
import transformers

from config import Config


def print(*args, **kwargs):
    if idr_torch.rank == 0 or kwargs.pop("force", False):
        bprint(*args, **kwargs)


def seed_all(seed: int):
    """
    Set the seed for all random number generators.
    """
    transformers.set_seed(seed)


def process_model_args(config: Config):
    model_id_or_path = config.mdlm_model_path
    ret = {
        "pretrained_model_name_or_path": model_id_or_path,
        "cache_dir": config.cache_dir,
    }
    if os.path.isdir(model_id_or_path):
        ret["local_files_only"] = True
    return ret


def get_tokenizer(config: Config, model: str):
    """
    Get the tokenizer from the config.
    """

    assert model in ["mdlm", "embedding"], f"model must be either 'mdlm' or 'embedding', got {model}"

    if model == "mdlm":
        path = config.mdlm_tokenizer

    elif model == "embedding":
        path = config.embedding_model if config.embedding_type == "external" else config.mdlm_model_path

    add_args = {"local_files_only": True} if os.path.isdir(path) else {}

    tokenizer = transformers.AutoTokenizer.from_pretrained(path, cache_dir=config.cache_dir, **add_args)

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


def compile_model(model, config: Config):
    """
    Compile the model using torch.compile
    """
    if config.compile_model:
        print("Compiling the model...")
        model = torch.compile(model)

    return model


def sample_categorical(categorical_probs: torch.Tensor, expand: int = None) -> torch.Tensor:
    if expand is not None:
        assert categorical_probs.dim() == 3, "categorical_probs must be of shape [B, T, V] to expand"
        categorical_probs = categorical_probs.repeat(expand, 1, 1)

    gumbel_norm: torch.Tensor = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


# Distributed utilities


class DistributedUtils:
    @classmethod
    def is_distributed(self) -> bool:
        return idr_torch.world_size > 1

    def __init__(self, cfg: Config):
        self.rank = idr_torch.rank
        self.local_rank = idr_torch.local_rank
        self.world_size = idr_torch.world_size
        self.cfg = cfg

        if self.is_distributed():
            self._setup_pg()

        # init the gather list
        self.init_placeholders()

    def init_placeholders(self):
        self.embeddings = torch.zeros(
            (self.world_size * self.cfg.batch_size, self.cfg.embedding_dim * self.cfg.sequence_length),
            device="cuda",
        )
        self.qualities = torch.zeros((self.world_size * self.cfg.batch_size,), device="cuda")

    def all_gather(
        self,
        local_embeddings: torch.Tensor,
        local_qualities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        assert self.is_distributed(), "all_gather can only be called in distributed mode"
        assert self.embeddings.is_cuda and self.qualities.is_cuda, "Placeholders must be on CUDA device"
        assert local_embeddings.is_cuda and local_qualities.is_cuda, "Local tensors must be on CUDA device"

        torch.distributed.all_gather_into_tensor(self.embeddings, local_embeddings)
        torch.distributed.all_gather_into_tensor(self.qualities, local_qualities)

        if self.rank != 0:
            return None, None

        return self.embeddings, self.qualities

    def dispatch_sequences(self, seq_ids: torch.Tensor | None, last: bool = False) -> torch.Tensor:
        assert self.is_distributed(), "dispatch_sequences can only be called in distributed mode"

        gather_indices = [None for _ in range(self.world_size)]

        if seq_ids is not None:
            seq_ids = seq_ids.to(dtype=torch.int32, device="cuda")

        torch.distributed.all_gather_object(gather_indices, seq_ids)

        all_indices_ = [idx.to("cuda") for idx in gather_indices if idx is not None]
        all_indices = torch.cat(all_indices_, dim=0)

        if last:
            return all_indices

        assert all_indices.size(0) == self.world_size * self.cfg.batch_size, "All indices size mismatch"

        rank_indices = all_indices[self.rank * self.cfg.batch_size : (self.rank + 1) * self.cfg.batch_size]
        return rank_indices

    def dispatch_batch_indices(self, ids: torch.Tensor | None) -> torch.Tensor | None:
        """
        Gather and slice batch indices across distributed processes.
        """
        assert self.is_distributed(), "dispatch_batch_indices can only be called in distributed mode"

        if ids is not None:
            ids = ids.to(dtype=torch.int16, device="cuda")

        gather_indices = [None for _ in range(self.world_size)]

        torch.distributed.all_gather_object(gather_indices, ids)
        all_indices_ = [idx for idx in gather_indices if idx is not None]
        all_indices = torch.cat(all_indices_, dim=0)

        assert all_indices.size(0) == self.world_size * self.cfg.n_groups, "All batch indices size mismatch"

        local_indices = self._get_local_indices(all_indices)
        return local_indices.to(dtype=torch.long, device="cuda") if local_indices is not None else None

    def _get_local_indices(self, global_indices: torch.Tensor) -> torch.Tensor | None:
        # get the indices for this rank

        mask = (global_indices >= self.rank * self.cfg.batch_size) & (
            global_indices < (self.rank + 1) * self.cfg.batch_size
        )
        local_indices = global_indices[mask]

        if local_indices.numel() == 0:
            return None

        local_indices = local_indices - self.rank * self.cfg.batch_size

        return local_indices

    def _setup_pg(self):
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(device)

    def cleanup(self):
        if not self.is_distributed():
            return
        torch.distributed.destroy_process_group()

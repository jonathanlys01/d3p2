import os
import warnings
from builtins import print as bprint
from typing import Iterable, TypeVar

import idr_torch
import numpy as np
import torch
import transformers
from idr_torch import IdrTorchWarning
from tqdm import tqdm as tqdm_

from config import Config


T = TypeVar("T")


def tqdm(it: Iterable[T], **kwargs) -> tqdm_:
    ret: tqdm_[T] = tqdm_(it, **kwargs)  # type: ignore
    return ret


warnings.filterwarnings("ignore", category=IdrTorchWarning)  # ignore idr_torch warnings


def print(*args, **kwargs):
    """Print only from rank 0."""
    if kwargs.pop("force", False) or idr_torch.rank == 0:
        bprint(*args, **kwargs)


def seed_all(seed: int):
    """
    Set the seed for all random number generators.
    """
    transformers.set_seed(seed)


def process_model_args(path, **kwargs):
    ret = dict(kwargs.items())
    ret["pretrained_model_name_or_path"] = path

    if os.path.isdir(path):
        ret["local_files_only"] = True
    return ret


def get_tokenizer(config: Config, model: str):
    """
    Get the tokenizer from the config.
    """

    assert model in ["mdlm", "llada"], f"Unknown model type: {model}"

    if model == "llada":
        path = config.llada_tokenizer
        return transformers.AutoTokenizer.from_pretrained(
            config.llada_model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            local_files_only=os.path.isdir(path),
        )

    path = config.mdlm_tokenizer
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


def sample_categorical_deprecated(categorical_probs: torch.Tensor, expand: int | None = None) -> torch.Tensor:
    """
    Gumbel-max trick for sampling from categorical distribution.
    categorical_probs: [B, T, V] tensor of categorical probabilities
    expand: if not None, number of samples to draw per input
    returns: [B, T] or [B*expand, T] tensor of sampled

    NB: This function is deprecated, the repeat_interleave operation is inefficient as it materializes large
    intermediate tensors on the GPU. Use sample_categorical instead.
    """
    if expand is not None:
        assert categorical_probs.dim() == 3, "categorical_probs must be of shape [B, T, V] to expand"
        categorical_probs = torch.repeat_interleave(categorical_probs, repeats=expand, dim=0)

    gumbel_norm: torch.Tensor = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def sample_categorical(categorical_probs: torch.Tensor, expand: int | None = None) -> torch.Tensor:
    expand = expand or 1
    B, T, V = categorical_probs.shape

    flat = categorical_probs.reshape(B * T, V)
    idx = torch.multinomial(flat, num_samples=expand, replacement=True)

    if expand == 1:
        return idx.view(B, T)

    return idx.view(B, T, expand).permute(0, 2, 1).reshape(B * expand, T)


# Distributed utilities


class DistributedUtils:
    """Utility class for distributed inference and data gathering."""

    @classmethod
    def is_distributed(self) -> bool:
        return idr_torch.world_size > 1  # type: ignore

    def __init__(self, cfg: Config):
        self.rank: int = idr_torch.rank  # type: ignore
        self.local_rank: int = idr_torch.local_rank  # type: ignore
        self.world_size: int = idr_torch.world_size  # type: ignore
        self.cfg = cfg

        if self.is_distributed():
            self._setup_pg()

        # init the gather list
        self.init_placeholders()

    def init_placeholders(self):
        if self.cfg.model == "mdlm":
            seq_len = self.cfg.sequence_length
        elif self.cfg.model == "llada":
            seq_len = self.cfg.block_length
        else:
            raise ValueError(f"Unknown model type: {self.cfg.model}")
        b_size = self.world_size * self.cfg.batch_size

        self.embeddings = torch.zeros((b_size, self.cfg.embedding_dim * seq_len), device="cuda")
        self.qualities = torch.zeros((b_size,), device="cuda")

        self.seq_ids_buffer = torch.zeros((b_size,), dtype=torch.int32, device="cuda")
        self.batch_indices_buffer = torch.zeros(
            (self.world_size * self.cfg.n_groups,),
            dtype=torch.int16,
            device="cuda",
        )

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

    def all_gather_scores(
        self,
        local_scores: torch.Tensor,
    ) -> torch.Tensor | None:
        assert self.is_distributed(), "all_gather_scores can only be called in distributed mode"
        assert self.qualities.is_cuda, "Placeholder must be on CUDA device"
        assert local_scores.is_cuda, "Local scores must be on CUDA device"

        torch.distributed.all_gather_into_tensor(self.qualities, local_scores)

        if self.rank != 0:
            return None

        return self.qualities

    def all_gather_embeddings(
        self,
        local_embeddings: torch.Tensor,
    ) -> torch.Tensor | None:
        assert self.is_distributed(), "all_gather_embeddings can only be called in distributed mode"
        assert self.embeddings.is_cuda, "Placeholder must be on CUDA device"
        assert local_embeddings.is_cuda, "Local embeddings must be on CUDA device"

        torch.distributed.all_gather_into_tensor(self.embeddings, local_embeddings)

        if self.rank != 0:
            return None

        return self.embeddings  # [B, L*E]

    def dispatch_sequences(self, seq_ids: torch.Tensor | None, last: bool = False) -> torch.Tensor:
        assert self.is_distributed(), "dispatch_sequences can only be called in distributed mode"

        # Determine local size
        local_size = seq_ids.size(0) if seq_ids is not None else 0

        # Gather sizes from all ranks
        sizes = torch.tensor([local_size], dtype=torch.int32, device="cuda")
        all_sizes = torch.zeros((self.world_size,), dtype=torch.int32, device="cuda")
        torch.distributed.all_gather_into_tensor(all_sizes, sizes)

        # Handle empty case
        total_size = all_sizes.sum().item()
        if total_size == 0:
            return torch.empty((0,), dtype=torch.int32, device="cuda")

        # Pad to max size and gather
        max_size: int = int(all_sizes.max().item())

        if local_size == 0 or seq_ids is None:
            local_data = torch.full((max_size,), -1, dtype=torch.int32, device="cuda")
        elif local_size < max_size:
            local_data = seq_ids.to(dtype=torch.int32, device="cuda")
            pad_size: int = max_size - local_size
            padding = torch.full((pad_size,), -1, dtype=torch.int32, device="cuda")
            local_data = torch.cat([local_data, padding], dim=0)
        else:
            local_data = seq_ids.to(dtype=torch.int32, device="cuda")

        # Gather all data
        buffer_size: int = self.world_size * max_size
        gather_buffer = torch.zeros((buffer_size,), dtype=torch.int32, device="cuda")
        torch.distributed.all_gather_into_tensor(gather_buffer, local_data)

        # Filter out sentinel values
        all_indices = gather_buffer[gather_buffer != -1]

        if last:
            return all_indices

        # Slice for this rank (matching original logic)
        rank_indices = all_indices[self.rank * self.cfg.batch_size : (self.rank + 1) * self.cfg.batch_size]
        return rank_indices

    def dispatch_batch_indices(self, ids: torch.Tensor | None) -> torch.Tensor | None:
        """
        Optimized gather and slice batch indices across distributed processes.
        Uses all_gather_into_tensor for much faster communication.
        """
        assert self.is_distributed(), "dispatch_batch_indices can only be called in distributed mode"

        # Determine local contribution size
        local_size = ids.size(0) if ids is not None else 0

        # Gather sizes from all ranks to know how much data each contributes
        sizes = torch.tensor([local_size], dtype=torch.int32, device="cuda")
        all_sizes = torch.zeros((self.world_size,), dtype=torch.int32, device="cuda")
        torch.distributed.all_gather_into_tensor(all_sizes, sizes)

        total_size = all_sizes.sum().item()
        if total_size == 0:
            return None

        # Prepare local data with padding if needed
        max_local_size: int = int(all_sizes.max().item())
        if ids is None:
            local_data = torch.full((max_local_size,), -1, dtype=torch.int16, device="cuda")
        else:
            local_data = ids.to(dtype=torch.int16, device="cuda")
            if local_data.size(0) < max_local_size:
                # Pad with sentinel value
                pad_size: int = max_local_size - local_data.size(0)
                padding = torch.full((pad_size,), -1, dtype=torch.int16, device="cuda")
                local_data = torch.cat([local_data, padding], dim=0)

        # Gather all data
        buffer_size: int = self.world_size * max_local_size
        gather_buffer = torch.zeros((buffer_size,), dtype=torch.int16, device="cuda")
        torch.distributed.all_gather_into_tensor(gather_buffer, local_data)

        # Filter out padding (-1 sentinel values)
        all_indices = gather_buffer[gather_buffer != -1]

        # Validate expected size
        expected_size = self.world_size * self.cfg.n_groups
        assert all_indices.size(0) == expected_size, (
            f"All batch indices size mismatch: {all_indices.size(0)} != {expected_size}"
        )

        # Get local indices for this rank
        local_indices = self._get_local_indices(all_indices.to(dtype=torch.long))
        return local_indices

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
        if not torch.distributed.is_initialized():
            print("Initializing process group")
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


# Dataloading utilities


def get_initial_data(tokenizer, mask_index: int, config: Config) -> torch.Tensor:
    """Get initial data for sampling based on the initial_mask_ratio in config."""

    path_to_bin = config.data_path
    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = config.sequence_length - 2  # account for bos/eos tokens
    start_idx = np.random.randint(0, len(data) - seq_length - 1, size=config.batch_size)
    batch_data = np.stack([data[i : i + seq_length] for i in start_idx], axis=0)
    bos_tensor = np.full((config.batch_size, 1), tokenizer.bos_token_id, dtype=np.int64)
    eos_tensor = np.full((config.batch_size, 1), tokenizer.eos_token_id, dtype=np.int64)
    batch_data = np.concatenate([bos_tensor, batch_data, eos_tensor], axis=1)
    batch_data = torch.from_numpy(batch_data).to(torch.int64)
    L = seq_length + 2
    num_tokens_to_mask = int(config.initial_mask_ratio * L)
    rand = torch.rand(config.batch_size, L)
    _, indices = torch.topk(rand, k=num_tokens_to_mask, dim=1)  # B x num_tokens_to_mask
    rows = torch.arange(config.batch_size).unsqueeze(1).expand(-1, num_tokens_to_mask)
    batch_data[rows, indices] = mask_index

    return batch_data

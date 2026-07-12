"""Small, single-process configuration for D5P4 LLaDA inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class D5P4Config:
    """Essential sampler and DPP-resampling settings."""

    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 1.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336

    n_groups: int = 2
    group_size: int = 2
    resample_start: int = 0
    resample_end: int | None = None

    kernel_type: str = "cosine"
    kernel_method: str = "multiplicative"
    quality_weight: float = 0.0
    rbf_gamma: float = 1.0
    score_method: str = "entropy"

    @property
    def batch_size(self) -> int:
        return self.n_groups * self.group_size

    @property
    def num_blocks(self) -> int:
        return self.gen_length // self.block_length

    @property
    def steps_per_block(self) -> int:
        return self.steps // self.num_blocks

    def validate(self) -> None:  # noqa: C901
        assert self.steps > 0
        assert self.gen_length > 0
        assert self.block_length > 0
        assert self.gen_length % self.block_length == 0
        assert self.steps % self.num_blocks == 0
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be 'low_confidence' or 'random'")
        if self.n_groups <= 0 or self.group_size <= 0:
            raise ValueError("n_groups and group_size must be positive")
        if self.resample_start < 0:
            raise ValueError("resample_start must be non-negative")
        if self.resample_end is not None and self.resample_end < self.resample_start:
            raise ValueError("resample_end must be greater than or equal to resample_start")
        if self.kernel_type not in {"cosine", "rbf"}:
            raise ValueError("kernel_type must be 'cosine' or 'rbf'")
        if self.kernel_method not in {"multiplicative", "additive"}:
            raise ValueError("kernel_method must be 'multiplicative' or 'additive'")
        if self.quality_weight < 0:
            raise ValueError("quality_weight must be non-negative")
        if self.rbf_gamma <= 0:
            raise ValueError("rbf_gamma must be positive")
        if self.score_method not in {"entropy", "mean_token_confidence"}:
            raise ValueError("score_method must be 'entropy' or 'mean_token_confidence'")

    def should_resample(self, step_in_block: int) -> bool:
        end = self.steps_per_block - 1 if self.resample_end is None else self.resample_end
        return self.resample_start <= step_in_block <= end

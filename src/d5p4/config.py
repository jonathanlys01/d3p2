import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import idr_torch
from omegaconf import OmegaConf


if TYPE_CHECKING:
    import torch

AVAIL = [
    "dpp",
    "exhaustive",
    "greedy_map",
    "greedy_beam",
    "diverse_beam",
    "random",
    "baseline",
    "ltr_beam",
    "_greedy_map",
]


SEQUENCE_LENGTH = 1_024
HIDDEN_SIZE_MDLM = 768
HIDDEN_SIZE_LLADA = 4_096
HIDDEN_SIZE_DREAM = 3_584
HIDDEN_SIZE_AR = 4_096
HIDDEN_SIZE_UDLM = 768
HIDDEN_SIZE_GIDD = 4_096
RESULTS_DIR = "results"
CACHE_DIR = "./.cache"
MODEL_EMBEDDING_DIMS = {
    "mdlm": HIDDEN_SIZE_MDLM,
    "llada": HIDDEN_SIZE_LLADA,
    "dream": HIDDEN_SIZE_DREAM,
    "udlm": HIDDEN_SIZE_UDLM,
    "gidd": HIDDEN_SIZE_GIDD,
    "ar": HIDDEN_SIZE_AR,
}
MODEL_CHOICES = set(MODEL_EMBEDDING_DIMS)
TIME_GRID_CHOICES = {"linear", "loglinear"}
POSTERIOR_SAMPLER_CHOICES = {"udlm_posterior", "gidd_posterior", "gidd_hf_generate"}
SCORE_METHOD_CHOICES = {
    "entropy",
    "self-certainty",
    "max_prob",
    "mean_token_confidence",
    "sequence_logprob",
    "delta_confidence",
}
GIDD_SCHEDULE_CHOICES = {"uniform", "hybrid"}
REMASKING_CHOICES = {"low_confidence", "selection_temperature", "random"}
LLADA_DECODER_CHOICES = {"diffusion", "classic_beam"}
DREAM_ALG_CHOICES = {"origin", "maskgit_plus", "topk_margin", "entropy"}
EVAL_SELECTION_METRIC_CHOICES = {"ppl", "f1", "int"}
CODE_DATASET_CHOICES = {"humaneval", "mbpp"}

CONFIG_FLAGS = ("--config", "-c", "config", "cfg")


def env_path_or(env_name: str, suffix: str, fallback: str) -> str:
    val = os.getenv(env_name)
    return str(Path(val) / suffix) if val else fallback


def get_user(default: str = "user") -> str:
    for var in ("USER", "LOGNAME", "USERNAME"):
        val = os.getenv(var)
        if val:
            return val
    return default


OmegaConf.register_new_resolver("env_path_or", env_path_or, replace=True)
OmegaConf.register_new_resolver("user", get_user, replace=True)


@dataclass(frozen=True)
class Config:
    disable_sys_args: bool = False

    sequence_length: int = SEQUENCE_LENGTH
    embedding_dim: int = 0  # to be set in __post_init__
    model: str = "mdlm"  # "mdlm", "llada", "dream", "udlm", "gidd", "ar"

    seed: int = 0
    n_runs: int = 16
    compile_model: bool = True

    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer: str = "gpt2"

    # LLaDA
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base"
    llada_tokenizer: str = "GSAI-ML/LLaDA-8B-Base"
    cfg_scale: float = 0.0
    llada_steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    # "diffusion" or forced left-to-right "classic_beam". With classic_beam, `transversal=true`
    # partitions the beam into n_groups groups of group_size, each searched independently.
    llada_decoder: str = "diffusion"
    classic_beam_branching_factor: int | None = None
    # Force the diffusion sampler to unmask strictly left-to-right (implicit for classic_beam).
    force_left_to_right: bool = False
    remasking: str = "low_confidence"  # "low_confidence", "selection_temperature", or "random"
    selection_temperature: float = 1.0
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = True
    guidance_start: int = 0  # step at which to start applying CFG (0-indexed)
    guidance_end: int = -1  # step at which to stop applying CFG (-1 means steps)

    # Dream
    dream_model_path: str = "Dream-org/Dream-v0-Instruct-7B"
    dream_tokenizer: str = "Dream-org/Dream-v0-Instruct-7B"
    dream_steps: int = 256
    dream_eps: float = 1e-3
    dream_alg: str = "entropy"
    dream_alg_temp: float | None = 0.0
    dream_top_p: float | None = 0.9
    dream_top_k: int | None = None

    # Autoregressive
    ar_model_path: str = "meta-llama/Meta-Llama-3-8B"
    ar_tokenizer: str = "meta-llama/Meta-Llama-3-8B"
    ar_embedding_method: str = "last"  # "last" or "mean" for AR embedding selection

    # UDLM / GIDD
    udlm_model_path: str = "kuleshov-group/udlm-lm1b"
    gidd_model_path: str = "dvruette/gidd-unif-3b"
    diffusion_steps: int = 128
    sampling_eps: float = 1e-5
    time_grid: str = "linear"  # "linear", "loglinear"
    posterior_sampler: str = "udlm_posterior"  # "udlm_posterior", "gidd_posterior", "gidd_hf_generate"
    self_correction: bool = False
    self_correction_temp: float = 0.1
    gidd_block_length: int = 128
    gidd_schedule: str = "uniform"  # "uniform", "hybrid"
    gidd_hybrid_p_unif: float = 1.0

    # sampling
    mdlm_steps: int = SEQUENCE_LENGTH  # number of MDLM sampling steps
    cat_temperature: float = 1.0

    # Source data
    data_path: str = "path_to.bin"
    initial_mask_ratio: float = 1.0  # ratio of tokens to mask at start of sampling (1.0 = all tokens masked)
    single_init: bool = True  # sample a single sequence and repeat it across the batch

    # Subset selection ###################################################################################
    method: str = "random"  # subset selection method
    transversal: bool = True  # use transversal sampling

    group_size: int = 2
    n_groups: int = 2

    # Subsample parameters (specific to each method)

    _kernel_type: str = "rbf"  # type of kernel to use in DPP
    _kernel_method: str = "additive"  # "additive": w*S + diag(q), "multiplicative": diag(q) @ S @ diag(q)
    _kernel_power: int = 1  # power for eigenvalue modulation
    _w_interaction: float = 0.0  # weight for diversity term in DPP, -1 for no quality term
    _w_split: float = 0.0  # weight for split groups in DPP
    _rbf_gamma: float = 1  # RBF kernel gamma parameter (when using RBF kernel)
    _temperature: float = 0.0  # temperature for any sampling
    _diversity_alpha: float = 0.0  # diversity coefficient for diverse beam search
    _score_method: str = "entropy"  # "entropy" or "self-certainty" (CE with uniform distribution)
    ######################################################################################################

    # windowing
    subsample_start: int = -1
    subsample_end: int = 1024
    subsample_k: int = 0  # if > 0 and < batch_size, subsample k best sequences from pool based on perplexity

    # eval
    eval_batch_size: int = 8  # batch size for evaluation (separate from inference batch_size)
    skip_eval: bool = False
    ppl_model_id: str = "gpt2"
    cos_model_id: str = "jinaai/jina-embeddings-v2-base-en"
    eval_selection_metric: str = "ppl"  # "ppl", "f1", or "int" for final sequence selection
    eval_transversal_group_representatives: bool = False  # pick one final representative per transversal group

    qa_dataset: str = "truthful_qa"  # "truthful_qa", "commonsense_qa", "ai2_arc", or "gsm8k"
    qa_dataset_len: int = -1  # number of samples to use from qa_dataset (-1 for all)
    qa_n_shots: int = 0  # number of few-shot examples for QA
    truthful_qa_path: str = "truthfulqa/truthful_qa"
    commonsense_qa_path: str = "tau/commonsense_qa"
    ai2_arc_path: str = "allenai/ai2_arc"
    ai2_arc_subset: str = "ARC-Challenge"
    gsm8k_path: str = "openai/gsm8k"

    # Code generation evaluation
    code_dataset: str = "humaneval"  # "humaneval" or "mbpp"
    code_dataset_len: int = -1  # number of samples to use from code_dataset (-1 for all)
    code_n_shots: int = 0
    humaneval_path: str = "openai/openai_humaneval"
    mbpp_path: str = "google-research-datasets/mbpp"
    mbpp_subset: str = "sanitized"
    code_timeout_s: float = 5.0

    # cache
    cache_dir: str = CACHE_DIR
    results_dir: str = RESULTS_DIR
    resume_runs: bool = True
    resume_db_dir: str | None = None
    resume_db_timeout_s: float = 60.0
    resume_db_keep_completed: bool = False
    # Reuse resume databases created before Dream was added to Config.
    legacy_config: bool = False

    # optuna
    n_trials: int = 100  # number of Optuna trials for hyperparameter sweeps
    comment: str = ""
    prompt: str | None = None

    batch_size: int = 0  # to be set in __post_init__
    interactive: bool = True
    minimal_log: bool = False
    quiet: bool = False
    standalone_job: bool = False  # ignore launcher-provided distributed metadata for this process

    def __post_init__(self):  # noqa: C901
        self._set_derived_fields()

        if not self.disable_sys_args:
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
            self._set_derived_fields()

        if self.n_runs == 1:
            object.__setattr__(self, "interactive", True)

        self._validate()

    def _set_derived_fields(self):
        if self.model not in MODEL_EMBEDDING_DIMS:
            raise ValueError(
                f"Model {self.model} not recognized. Available models: {sorted(MODEL_CHOICES)}",
            )
        if self.model == "gidd" and self.posterior_sampler == "udlm_posterior":
            object.__setattr__(self, "posterior_sampler", "gidd_hf_generate")
        object.__setattr__(self, "embedding_dim", MODEL_EMBEDDING_DIMS[self.model])
        object.__setattr__(self, "batch_size", self.n_groups * self.group_size)

    def _validate(self):
        assert self.method in AVAIL, f"Method {self.method} not recognized. Available methods: {list(AVAIL)}"
        assert self.model in MODEL_CHOICES, (
            f"Model {self.model} not recognized. Available models: {sorted(MODEL_CHOICES)}"
        )
        if self.method == "ltr_beam":
            assert self.model == "llada" and self.llada_decoder == "classic_beam", (
                "method=ltr_beam requires model=llada and llada_decoder=classic_beam"
            )
        assert 0 < self.initial_mask_ratio <= 1.0, "initial_mask_ratio must be in (0, 1]"
        assert self.eval_selection_metric in EVAL_SELECTION_METRIC_CHOICES, (
            f"eval_selection_metric must be one of {sorted(EVAL_SELECTION_METRIC_CHOICES)}, "
            f"got {self.eval_selection_metric!r}"
        )
        assert self.code_dataset in CODE_DATASET_CHOICES, (
            f"code_dataset must be one of {sorted(CODE_DATASET_CHOICES)}, got {self.code_dataset!r}"
        )
        assert self.code_n_shots >= 0, "code_n_shots must be non-negative"
        assert self.code_timeout_s > 0.0, "code_timeout_s must be positive"

        if self.subsample_k > 0:
            assert self.method == "baseline", "subsample_k only makes sense for baseline method"

        assert self.diffusion_steps > 0, "diffusion_steps must be positive"
        assert 0.0 < self.sampling_eps < 1.0, "sampling_eps must be in (0, 1)"
        assert self.time_grid in TIME_GRID_CHOICES, f"Unknown time_grid: {self.time_grid}"
        assert self.posterior_sampler in POSTERIOR_SAMPLER_CHOICES, (
            f"Unknown posterior_sampler: {self.posterior_sampler}"
        )
        assert self._score_method in SCORE_METHOD_CHOICES, f"Unknown _score_method: {self._score_method}"
        assert self.gidd_block_length > 0, "gidd_block_length must be positive"
        assert self.gidd_schedule in GIDD_SCHEDULE_CHOICES, f"Unknown gidd_schedule: {self.gidd_schedule}"
        assert 0.0 <= self.gidd_hybrid_p_unif <= 1.0, "gidd_hybrid_p_unif must be in [0, 1]"

        if self.model == "llada":
            self._validate_llada()
        elif self.model == "dream":
            self._validate_dream()

    def _validate_llada(self):
        assert self.llada_decoder in LLADA_DECODER_CHOICES, (
            f"llada_decoder must be one of {sorted(LLADA_DECODER_CHOICES)}, got {self.llada_decoder!r}"
        )
        if self.classic_beam_branching_factor is not None:
            assert self.classic_beam_branching_factor > 0, "classic_beam_branching_factor must be positive"
        if self.llada_decoder == "classic_beam":
            assert self.cfg_scale == 1.0, "classic_beam requires conditional-only cfg_scale=1.0"
            assert self.method == "ltr_beam", "classic_beam requires method=ltr_beam"
            assert not self.logits_eos_inf, "classic_beam requires logits_eos_inf=false so beams can terminate"
            assert not self.force_left_to_right, "classic_beam is already left-to-right; unset force_left_to_right"
            # beam_size comes from n_groups * group_size; beam search at width 1 is greedy decoding,
            # which is almost never what was intended and is otherwise silent.
            assert self.batch_size > 1, "classic_beam requires batch_size=n_groups*group_size > 1, got 1"
            if self.transversal:
                # Partitioned beam search: n_groups independent beams of group_size, the same
                # partition the D5P4 arm gets from transversal selection.
                assert self.n_groups > 1, "transversal classic_beam needs n_groups > 1"
                assert self.group_size > 1, (
                    "transversal classic_beam with group_size=1 is n_groups independent greedy "
                    "chains; set group_size > 1 for a real beam per group"
                )
            else:
                assert self.group_size == 1, (
                    "non-transversal classic_beam requires group_size=1; "
                    "encode the full global beam width in n_groups"
                )

        assert self.remasking in REMASKING_CHOICES, f"Remasking method {self.remasking} not recognized."
        if self.eval_transversal_group_representatives:
            assert self.transversal, "eval_transversal_group_representatives requires transversal=True"
            assert self.group_size > 1, "eval_transversal_group_representatives requires group_size > 1"
        assert self.selection_temperature >= 0.0, "selection_temperature must be non-negative"
        if self.llada_decoder == "diffusion":
            assert self.gen_length % self.block_length == 0, "gen_length must be divisible by block_length"
            num_blocks = self.gen_length // self.block_length
            assert self.llada_steps % num_blocks == 0, "llada_steps must be divisible by num_blocks"
        if self.force_left_to_right and idr_torch.rank == 0:
            inert = [
                name
                for name, is_set in (
                    (f"remasking={self.remasking}", self.remasking != "low_confidence"),
                    ("selection_temperature", self.remasking == "selection_temperature"),
                    ("confidence_eos_eot_inf", self.confidence_eos_eot_inf),
                )
                if is_set
            ]
            if inert:
                print(
                    f"Warning: force_left_to_right makes {', '.join(inert)} inert; "
                    "the unmasking order is the position order, not a confidence ranking.",
                )
        elif self.remasking == "selection_temperature" and self.cat_temperature != 0.0 and idr_torch.rank == 0:
            print(
                "Warning: remasking=selection_temperature with cat_temperature != 0.0. "
                "This mixes stochastic token sampling with stochastic remasking.",
            )

        if self.guidance_end == -1:
            object.__setattr__(self, "guidance_end", self.llada_steps)

        assert 0 <= self.guidance_start < self.guidance_end, (
            f"guidance_start ({self.guidance_start}) must be >= 0 and < guidance_end ({self.guidance_end})"
        )
        assert self.guidance_end <= self.llada_steps, (
            f"guidance_end ({self.guidance_end}) must be <= llada_steps ({self.llada_steps})"
        )

    def _validate_dream(self):
        assert self.dream_steps > 0, "dream_steps must be positive"
        assert 0.0 < self.dream_eps < 1.0, "dream_eps must be in (0, 1)"
        assert self.dream_alg in DREAM_ALG_CHOICES, (
            f"dream_alg must be one of {sorted(DREAM_ALG_CHOICES)}, got {self.dream_alg!r}"
        )
        assert self.cat_temperature >= 0.0, "cat_temperature must be non-negative"
        if self.dream_alg_temp is not None:
            assert self.dream_alg_temp >= 0.0, "dream_alg_temp must be non-negative"
        if self.dream_top_p is not None:
            assert 0.0 < self.dream_top_p <= 1.0, "dream_top_p must be in (0, 1]"
        if self.dream_top_k is not None:
            assert self.dream_top_k > 0, "dream_top_k must be positive"

    def __str__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self))


@dataclass
class Cache:
    """Per-step state handed to subsample selectors.

    All tensors are sliced to the current block: shapes are [B, block_length(, ...)].
    `log_p_x0` may be a non-contiguous view; `embeddings` is None when the selector
    declares `needs_embeddings = False` (see BaseSelector).
    """

    x: Optional["torch.Tensor"] = None
    log_p_x0: Optional["torch.Tensor"] = None
    embeddings: Optional["torch.Tensor"] = None


# Utils
def _expand_path(value: str) -> str:
    """Expand environment variables and user home in a path string."""
    return os.path.expandvars(os.path.expanduser(value))


def _is_likely_path(value: str) -> bool:
    """Check if a string is likely to be a path using multiple heuristics."""
    if not isinstance(value, str) or not value:
        return False

    # Check if it's an existing directory (definitive proof it's a path)
    if os.path.isdir(_expand_path(value)):
        return True

    if "/" in value or "\\" in value:
        return True
    path_extensions = (".bin", ".pt", ".pth", ".yaml", ".yml", ".json", ".txt", ".csv", ".log")
    if any(value.endswith(ext) for ext in path_extensions):
        return True
    path_keywords = ("path", "dir", "file", "cache", "results")
    return any(kw in value.lower() for kw in path_keywords)


if __name__ == "__main__":
    config = Config()
    config_dict = OmegaConf.to_container(OmegaConf.structured(config))
    statuses = ["\033[91m(NOK)\033[0m", "\033[92m(OK)\033[0m"]

    print("Verifying paths...")
    print("=" * 50)

    assert hasattr(config_dict, "items")
    for key, value in config_dict.items():
        if isinstance(value, str) and _is_likely_path(value):
            exists = os.path.exists(_expand_path(value))
            print(f"{key}: {value} {statuses[exists]}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 50)

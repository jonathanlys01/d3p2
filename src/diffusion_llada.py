"""
Minimalist LLaDA diffusion sampler, adapted from the LLaDA codebase
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from config import Cache, Config
from data import truthful_qa
from llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from subsample import get_subsample_selector
from utils import get_tokenizer, process_model_args, sample_categorical


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Kept for compatibility, but not used in pure diffusion (greedy) sampling.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log(noise)
    return (logits / temperature).exp() / gumbel_noise


class LLADASampler(nn.Module):
    """Discrete Diffusion Model base class. (LLaDA version)"""

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype="auto")
        self.model = LLaDAModelLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config
        self.tokenizer = get_tokenizer(config, "llada")

        model_config: LLaDAConfig = self.model.config
        self.mask_index = model_config.mask_token_id
        sequence_length = config.sequence_length
        assert sequence_length <= model_config.max_sequence_length, "Requested sequence length exceeds model's maximum."
        self.sequence_length = sequence_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states

            print("a" * 50, embeddings[-1].shape)
        return logits, embeddings

    def _sample_prior(self, *batch_dims) -> torch.Tensor:
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _get_num_transfer_tokens(self, t: int, prompt_length: int) -> torch.Tensor:
        T = self.config.num_steps
        # Calculate tokens based on the part we actually want to generate
        gen_len = self.sequence_length - prompt_length
        frac = (T - t) / T

        # How many NEW tokens should be visible?
        num_gen_tokens = torch.tensor(gen_len * frac, device=self.device, dtype=torch.int64)

        # Total visible = Prompt + New Tokens
        return num_gen_tokens.repeat(self.config.batch_size) + prompt_length

    def _update(
        self,
        x_t: torch.Tensor,
        t: int,
        remasking="confidence",
        cfg_scale: float = 0.0,
        prompt_length=0,
    ) -> torch.Tensor:
        if cfg_scale > 0.0:
            un_x = x_t.clone()
            un_x[:, :prompt_length] = self.mask_index
            x_ = torch.cat([x_t, un_x], dim=0)
            logits, out_all = self._forward_model(x_)
            embeddings_all = out_all[-1]

            logits, un_logits = torch.chunk(logits, 2, dim=0)
            embeddings, _ = torch.chunk(embeddings_all, 2, dim=0)  # we only need embeddings for the main batch

            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits, out = self._forward_model(x_t)
            embeddings = out[-1]
        cache = Cache(log_p_x0=logits, embeddings=embeddings, x=x_t)

        subsample_step = self.config.subsample_start <= t <= self.config.subsample_end
        last_step = t == -1

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(x_t.size(0), device=self.device)
        )

        if slice_idx is None:
            ret = None

        else:
            x_t = x_t[slice_idx]
            logits = logits[slice_idx]

            x0 = sample_categorical(logits, expand=self.config.group_size if subsample_step else None)
            print("b" * 50, x0.shape)

            # 3. Confidence Calculation
            if remasking == "confidence":
                p = F.softmax(logits, dim=-1)
                # Gather confidence of the PREDICTED tokens
                conf_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
            elif remasking == "random":
                conf_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 4. Prompt Preservation
            # Force prompt confidence to infinity so they are always kept in the top-k selection
            conf_p[:, :prompt_length] = float("inf")

            # 5. Masking Schedule (Linear)
            # Determine which tokens to keep (unmask) for the next step.
            # num_transfer_tokens is the TARGET TOTAL count of unmasked tokens.
            num_transfer_tokens = self._get_num_transfer_tokens(t, prompt_length)  # (B,)

            # We start with a full mask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(conf_p.shape[0] if "confidence" in locals() else x0.shape[0]):
                k = num_transfer_tokens[j].item()
                # Ensure we keep at least the prompt tokens
                k = max(k, prompt_length)

                # Select the top-k most confident tokens from the ENTIRE sequence
                _, select_index = torch.topk(conf_p[j], k=k)
                transfer_index[j, select_index] = True

            # 6. Update State
            # Where transfer_index is True, we keep the prediction x0.
            # Where transfer_index is False, we apply the mask token.
            x_next = torch.where(transfer_index, x0, torch.full_like(x0, self.mask_index))

            # Explicitly enforce prompt consistency (though infinite confidence should handle this)
            x_next[:, :prompt_length] = x_t[:, :prompt_length]

            ret = x_next

        return ret

    def _gen_prompt(self, prompt: str) -> torch.Tensor:
        if "instruct" in self.config.llada_model_path.lower():
            message = {"role": "user", "content": prompt}
            prompt = self.tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        prompt_tokens: torch.Tensor = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        init_x = self._sample_prior(self.config.batch_size, self.sequence_length)
        prompt_length = prompt_tokens.shape[1]
        init_x[:, :prompt_length] = prompt_tokens
        return init_x

    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def sample(
        self,
        num_steps: Optional[int] = None,
        init_x: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        num_steps = num_steps or self.config.num_steps
        prompt_length = 0

        if prompt is not None:
            assert init_x is None, "Cannot provide both prompt and init_x."
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
            init_x = self._sample_prior(self.config.batch_size, self.sequence_length)
            prompt_length = prompt_tokens.shape[1]
            init_x[:, :prompt_length] = prompt_tokens

        if init_x is None:
            init_x = self._sample_prior(self.config.batch_size, self.sequence_length)

        x_t = init_x.to(self.device)

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0

        for t in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps, disable=disable):
            x_t = self._update(
                x_t,
                t,
                prompt_length=prompt_length,
                cfg_scale=cfg_scale,
            )

        return x_t


def main():
    cfg = Config()
    sampler = LLADASampler(cfg)
    dataset = truthful_qa(cfg)

    samples = []
    prompts = []

    for row in dataset.itertuples():
        prompt = row.question + "\nAnswer:"

        samples.extend(sampler.sample(prompt=prompt, cfg_scale=cfg.cfg_scale))
        prompts.extend([prompt] * cfg.batch_size)

    with open("llada_min_truth_qa_samples.txt", "w") as f:
        for i, sample in enumerate(samples):
            decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=False)
            decoded_text = decoded_text.split("<|endoftext|>")[0]  # take content before EOS token
            f.write(f"Prompt: {prompts[i]}\n")
            f.write(f"Sample: {decoded_text}\n\n")
            f.write("=" * 80 + "\n\n")

    print("Done")


if __name__ == "__main__":
    main()

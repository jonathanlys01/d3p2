r"""
Minimalist LLaDA diffusion sampler, adapted from the LLaDA codebase

python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.5 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.6 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.7 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.8 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.9 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=1.0 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=1.5
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
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM,
    low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
    # gumbel_noise = -torch.log(noise)
    # return (logits / temperature).exp() / gumbel_noise


class LLADASampler(nn.Module):
    """Discrete Diffusion Model base class. (LLaDA version)"""

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype="auto")
        self.model = LLaDAModelLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config: Config = config
        self.tokenizer = get_tokenizer(config, "llada")

        model_config: LLaDAConfig = self.model.config
        self.mask_index = model_config.mask_token_id
        sequence_length = config.sequence_length
        assert sequence_length <= model_config.max_sequence_length, "Requested sequence length exceeds model's maximum."
        self.sequence_length = sequence_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):  # type: ignore
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states
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
        cfg_scale: float,
        remasking="confidence",
        prompt_length=0,
    ) -> torch.Tensor | None:
        if cfg_scale > 0.0:
            un_x = x_t.clone()
            un_x[:, :prompt_length] = self.mask_index
            x_ = torch.cat([x_t, un_x], dim=0)
            logits, out_all = self._forward_model(x_)
            embeddings_all = out_all[-1]

            logits, un_logits = torch.chunk(logits, 2, dim=0)
            _, embeddings = torch.chunk(embeddings_all, 2, dim=0)  # ignore conditional embeddings

            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits, out = self._forward_model(x_t)
            embeddings = out[-1]

        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
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
            # logp_x0 = logits[slice_idx]
            logp_x0 = torch.index_select(logits, 0, slice_idx)
            p_x0 = torch.exp(logp_x0)

            x_t = x_t[slice_idx]

            x0 = sample_categorical(p_x0, expand=self.config.group_size if subsample_step else None)

            # 3. Confidence Calculation
            if remasking == "confidence":
                expanded_p_x0 = p_x0.repeat_interleave(self.config.group_size, dim=0) if subsample_step else p_x0
                conf_p = torch.gather(expanded_p_x0, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
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
                k = int(num_transfer_tokens[j].item())
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
            x_t_expand = x_t.repeat_interleave(self.config.group_size, dim=0) if subsample_step else x_t
            x_next[:, :prompt_length] = x_t_expand[:, :prompt_length]

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
        cfg_scale: float,
        num_steps: Optional[int] = None,
        init_x: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
    ) -> torch.Tensor | None:
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

    def _get_block_transfer_tokens(self, mask_index, steps):
        """
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        """Apply chat template if needed, and tokenize the prompt."""
        if "instruct" in self.config.llada_model_path.lower():
            message = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = prompt

        encoded_outputs = self.tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_tokens = encoded_outputs["input_ids"].to(self.device)
        return prompt_tokens

    def _get_slice(self, t: int, cache: Cache) -> tuple[bool, torch.Tensor | None]:
        subsample_step = self.config.subsample_start <= t <= self.config.subsample_end
        last_step = t == -1

        assert cache.x is not None

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(cache.x.size(0), device=self.device)
        )

        return subsample_step, slice_idx

    def _block_sample(self, logits: torch.Tensor, subsample_step: bool) -> torch.Tensor:
        temperature = self.config.cat_temperature
        expand = self.config.group_size if subsample_step else 1

        if temperature == 0.0:
            x0_ = torch.argmax(logits, dim=-1)
            x0 = torch.repeat_interleave(x0_, repeats=expand, dim=0)
        else:
            logits = logits.to(torch.float64) / temperature
            probs = F.softmax(logits, dim=-1)
            x0 = sample_categorical(probs, expand=expand)
        return x0

    def _get_confidence(self, logits: torch.Tensor, x0: torch.Tensor, num_block: int, prompt_len: int) -> torch.Tensor:
        if self.config.confidence_eos_eot_inf:
            logits[:, :, 126348] = -torch.inf

        if self.config.remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
        elif self.config.remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise ValueError(f"Invalid remasking method: {self.config.remasking}")

        x0_p[:, prompt_len + (num_block + 1) * self.config.block_length :] = -torch.inf
        return x0_p

    @torch.no_grad()
    def block_diffuse(self, prompt: str):
        num_blocks = self.config.gen_length // self.config.block_length
        steps = self.config.steps // num_blocks
        batch_size = self.config.batch_size

        prompt_tokens = self._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        prompt_tokens = prompt_tokens.repeat(batch_size, 1)

        # Setup generation buffer
        x = torch.full(
            (batch_size, prompt_len + self.config.gen_length),
            self.mask_index,
            dtype=torch.long,
        ).to(self.device)
        x[:, :prompt_len] = prompt_tokens.clone()

        prompt_index = x != self.mask_index

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0

        for num_block in tqdm(range(num_blocks), desc="Blocks", disable=disable):
            start = prompt_len + num_block * self.config.block_length
            end = prompt_len + (num_block + 1) * self.config.block_length
            block_mask_index = x[:, start:end] == self.mask_index

            num_transfer_tokens = self._get_block_transfer_tokens(block_mask_index, steps)

            for step in range(steps):
                mask_index = x == self.mask_index

                if self.config.cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_index
                    x_ = torch.cat([x, un_x], dim=0)

                    logits, out_all = self._forward_model(x_)
                    embeddings_all = out_all[-1]

                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    embeddings, _ = torch.chunk(embeddings_all, 2, dim=0)

                    logits = un_logits + (self.config.cfg_scale + 1) * (logits - un_logits)
                else:
                    logits, out = self._forward_model(x)
                    embeddings = out[-1]

                if self.config.logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                cache = Cache(
                    log_p_x0=logits[:, start:end],
                    embeddings=embeddings[:, start:end],
                    x=x[:, start:end],
                )
                subsample_step, slice_idx = self._get_slice(step, cache)

                assert slice_idx is not None

                x0 = self._block_sample(torch.index_select(logits, 0, slice_idx), subsample_step)
                x0_p = self._get_confidence(logits, x0, num_block, prompt_len)

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(batch_size):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, step])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x


def main_block():
    limit = 50
    cfg = Config()
    sampler = LLADASampler(cfg)
    dataset = truthful_qa(cfg)

    samples = []
    prompts = []

    for i, row in enumerate(dataset.itertuples()):
        if i >= limit:
            break

        prompt = str(row.question)

        # sample using the block_diffuse method
        samples.extend(sampler.block_diffuse(prompt=prompt))
        prompts.extend([prompt] * cfg.batch_size)

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()

    with open(f"llada_block_{cfg.cfg_scale}.log", "w") as f:
        for i, sample in enumerate(samples):
            decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=False)
            f.write(f"{decoded_text}\n\n")
            f.write("=" * 80 + "\n\n")

    print("Done")


def main():
    limit = 1
    cfg = Config()
    sampler = LLADASampler(cfg)
    dataset = truthful_qa(cfg)

    samples = []
    prompts = []

    for i, row in enumerate(dataset.itertuples()):
        if i >= limit:
            break
        prompt = row.question

        samples.extend(sampler.sample(prompt=prompt, cfg_scale=cfg.cfg_scale))
        prompts.extend([prompt] * cfg.batch_size)

    if sampler.distributed_utils:
        # cleanup
        sampler.distributed_utils.cleanup()

    with open("llada_min_truth_qa_samples.log", "w") as f:
        for i, sample in enumerate(samples):
            decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=False)
            # decoded_text = decoded_text.split("<|endoftext|>")[0]  # take content before EOS token
            f.write(f"Prompt: {prompts[i]}\n")
            f.write(f"Sample: {decoded_text}\n\n")
            f.write("=" * 80 + "\n\n")

    print("Done")


if __name__ == "__main__":
    # main()
    main_block()

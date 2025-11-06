"""
Minimalist LLaDA diffusion sampler, adapted from the LLaDA codebase
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from config import Config
from dpp import SubsetSelector
from llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from utils import get_tokenizer, process_model_args


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class LLADASampler(nn.Module):
    """Discrete Diffusion Model base class. (LLaDA version)"""

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype="auto")
        self.model = LLaDAModelLM.from_pretrained(**model_args)
        self.selector = SubsetSelector(config)
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
        return logits, embeddings

    def _sample_prior(self, *batch_dims) -> torch.Tensor:
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _get_num_transfer_tokens(self, t: int) -> torch.Tensor:
        T = self.config.num_steps
        total_tokens = self.sequence_length
        frac = (T - t) / T
        num_tokens = torch.tensor(total_tokens * frac, device=self.device, dtype=torch.int64)
        return num_tokens.repeat(self.config.batch_size)

    def _update(self, x_t: torch.Tensor, t: int, remasking="confidence", temperature=0.0) -> torch.Tensor:
        logits, _ = self._forward_model(x_t)

        if t > self.config.num_steps - 10:  # first steps -> increase temperature
            remasking = "random"
            temperature = 1.0

        logits[:, :, 126081] = -torch.inf  # EOS token penalization
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # B, L

        if remasking == "confidence":
            p = F.softmax(logits, dim=-1)
            conf_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

        elif remasking == "random":
            conf_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        is_mask = x_t == self.mask_index
        x0 = torch.where(is_mask, x0, x_t)
        confidence = torch.where(is_mask, conf_p, -torch.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        num_transfer_tokens = self._get_num_transfer_tokens(t)
        for j in range(confidence.shape[0]):
            k = num_transfer_tokens[j].item()
            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True

        x0 = torch.where(transfer_index, x0, x_t)

        return x0

    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def sample(
        self,
        num_steps: Optional[int] = None,
        init_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_steps = num_steps or self.config.num_steps

        if init_x is None:
            init_x = self._sample_prior(self.config.batch_size, self.sequence_length)

        x_t = init_x.to(self.device)

        for t in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps):
            x_t = self._update(x_t, t)

        return x_t


def main():
    cfg = Config()

    # Create sampler and print a few basic attributes to verify initialization
    sampler = LLADASampler(cfg)

    samples = sampler.sample(init_x=None)

    for i, sample in enumerate(samples):
        decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "__")
        print(f"\nSample {i + 1}:\n{decoded_text}\n")

    return

    print("LLADASampler initialized.")
    print(" device:", sampler.device)
    print(" sequence_length:", sampler.sequence_length)
    print(" mask_index:", sampler.mask_index)
    print("dtype of model parameters:", next(sampler.model.parameters()).dtype)
    print(sampler.model)
    total_params = sum(p.numel() for p in sampler.model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    text = torch.tensor(
        sampler.tokenizer(
            "One avenue for addressing these issues is mechanistic interpretability, attempting to reverse engineer the detailed computations performed by transformers, similar to how a programmer might try to reverse engineer complicated binaries into human-readable source code.",  # noqa
        )["input_ids"],
    ).to(sampler.device)

    idx = len(text) // 2
    text[idx] = sampler.mask_index  # Introduce a mask token for testing

    print("Input text token IDs:", text)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = sampler.model.forward(text[None, :], return_dict=True)

    print("Model output logits shape:", output.logits.shape)
    print("LLADASampler test completed successfully.")
    print("Output logits for the input text:", output.logits)

    # check if non-mask tokens are argmax
    predicted_tokens = torch.argmax(output.logits, dim=-1)[0]
    for i in range(len(text)):
        if text[i] != sampler.mask_index:
            GREEN = "\033[32m"
            CYAN = "\033[36m"
            RESET = "\033[0m"
            orig_tok = sampler.tokenizer.decode([text[i].item()])
            pred_tok = sampler.tokenizer.decode([predicted_tokens[i].item()])
            print(
                f"Token position {i}: original token ID = {text[i].item()}, "
                f"with token: {GREEN}{orig_tok}{RESET}, "
                f"predicted token ID = {predicted_tokens[i].item()}, "
                f"with token: {CYAN}{pred_tok}{RESET}",
            )

    # decode top 3 tokens at the masked position
    masked_logits = output.logits[0, idx]
    top3_probs, top3_indices = torch.topk(torch.softmax(masked_logits, dim=-1), k=3)
    top3_tokens = [sampler.tokenizer.decode([idx.item()]) for idx in top3_indices]
    print("Top 3 predicted tokens at the masked position:")
    for token, prob in zip(top3_tokens, top3_probs):
        print(f" Token: '{token}' with probability {prob.item():.4f}")


if __name__ == "__main__":
    main()

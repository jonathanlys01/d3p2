"""
Autoregressive sampler with beam-style exploration.
Mimics the behavior of diffusion samplers but uses standard left-to-right generation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from config import Cache, Config
from subsample import get_subsample_selector
from utils import get_tokenizer, print, process_model_args, sample_categorical, tqdm


NEG_INFINITY = -1_000_000.0
torch.set_float32_matmul_precision("high")


class AutoregressiveSampler(nn.Module):
    """Autoregressive sampler with beam-style exploration."""

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.ar_model_path, cache_dir=config.cache_dir)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config

        # Use ar_tokenizer if specified, otherwise fall back to model path
        tokenizer_path = config.ar_tokenizer or config.ar_model_path
        if "gpt2" in tokenizer_path.lower():
            self.tokenizer = get_tokenizer(config, "mdlm")
        else:
            tokenizer_args = process_model_args(tokenizer_path, cache_dir=config.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

        self.model_length = config.gen_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def _prepend_bos(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prepend BOS token if not already present."""
        if tokens[0, 0] == self.tokenizer.bos_token_id:
            return tokens
        bos = torch.full((1, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
        return torch.cat([bos, tokens], dim=1)

    @torch.no_grad()
    def sample(self, prompt: str | None = None):  # noqa: C901, PLR0912, PLR0915
        batch_size = self.config.batch_size

        # Initialize sequence with prompt or BOS
        if prompt is not None:
            encoded = self.tokenizer([prompt], add_special_tokens=True, padding=False, return_tensors="pt")
            prompt_tokens = self._prepend_bos(encoded["input_ids"].to(self.device))
            seq = prompt_tokens.repeat(batch_size, 1)
            prompt_len = prompt_tokens.shape[1]
        else:
            seq = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
            prompt_len = 0

        attention_mask = torch.ones_like(seq, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        past_key_values = None

        # For mean embedding strategy: track cumulative embeddings
        use_mean_embedding = self.config.ar_embedding_method == "mean"
        embedding_sum: torch.Tensor | None = None
        embedding_count = 0

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0

        for step in tqdm(range(self.model_length), desc="Generating", disable=disable):
            subsample_step = self.config.subsample_start <= step <= self.config.subsample_end
            input_ids = seq if past_key_values is None else seq[:, -1:]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values
            embeddings = outputs.hidden_states[-1] if outputs.hidden_states else None
            logits = outputs.logits[:, -1]
            probs = F.softmax(logits, dim=-1).unsqueeze(1)

            # Update cumulative embedding sum for mean strategy
            if use_mean_embedding and embeddings is not None:
                current_emb = embeddings[:, -1, :]  # [B, hidden_dim]
                embedding_sum = current_emb.clone() if embedding_sum is None else embedding_sum + current_emb
                embedding_count += 1

            if subsample_step and self.config.group_size > 1:
                # Compute embedding for Cache based on strategy (always shape [B, 1, E])
                if use_mean_embedding and embedding_sum is not None:
                    mean_emb = (embedding_sum / embedding_count).unsqueeze(1)  # [B, 1, E]
                    cache_embeddings = mean_emb
                else:
                    cache_embeddings = embeddings[:, -1:] if embeddings is not None else None

                cache = Cache(
                    log_p_x0=logits.unsqueeze(1),
                    embeddings=cache_embeddings,
                    x=seq,
                )
                slice_idx = self.selector.subsample(cache)

                if slice_idx is not None:
                    # Slice state
                    probs = probs[slice_idx]
                    seq = seq[slice_idx]
                    finished = finished[slice_idx]
                    attention_mask = attention_mask[slice_idx]
                    if use_mean_embedding and embedding_sum is not None:
                        embedding_sum = embedding_sum[slice_idx]

                    # Expand
                    g = self.config.group_size
                    next_token = sample_categorical(probs, expand=g)
                    seq = seq.repeat_interleave(g, dim=0)
                    finished = finished.repeat_interleave(g, dim=0)
                    attention_mask = attention_mask.repeat_interleave(g, dim=0)
                    if use_mean_embedding and embedding_sum is not None:
                        embedding_sum = embedding_sum.repeat_interleave(g, dim=0)
                    if past_key_values is not None:
                        past_key_values = DynamicCache(
                            tuple(kv[slice_idx].repeat_interleave(g, dim=0) for kv in layer)
                            for layer in past_key_values
                        )
                else:
                    next_token = sample_categorical(probs, expand=None)
            else:
                next_token = sample_categorical(probs, expand=None)

            next_token[finished] = self.tokenizer.eos_token_id
            finished = finished | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            seq = torch.cat([seq, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, (~finished).long().unsqueeze(-1)], dim=1)

            if finished.all():
                break

        # Gather all sequences in distributed mode (handles variable lengths)
        if self.distributed_utils:
            seq, lengths = self.distributed_utils.all_gather_sequences_varlen(seq, self.tokenizer.pad_token_id)

        return seq, prompt_len


def main():
    config = Config(
        disable_sys_args=True,
        model="ar",
        batch_size=4,
        n_groups=2,
        group_size=2,
        method="greedy_map",
        transversal=True,
        gen_length=50,
    )

    print("Initializing AutoregressiveSampler...")
    sampler = AutoregressiveSampler(config)

    print(f"Model max length: {sampler.model_length}")
    print(f"BOS token: {sampler.tokenizer.bos_token_id}")
    print(f"EOS token: {sampler.tokenizer.eos_token_id}")
    print(f"PAD token: {sampler.tokenizer.pad_token_id}")

    print("\nGenerating samples...")
    sequences, prompt_len = sampler.sample()

    print(f"\nGenerated {sequences.shape[0]} sequences")
    print(f"Sequence shape: {sequences.shape}")

    print("\n" + "=" * 80)
    for i, seq in enumerate(sequences):
        eos_positions = (seq == sampler.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            after_eos = seq[first_eos + 1 :]
            if len(after_eos) > 0:
                all_eos = (after_eos == sampler.tokenizer.eos_token_id).all()
                print(f"\nSeq {i}: length={len(seq)}, EOS at pos {first_eos}, all EOS after: {all_eos}")
            else:
                print(f"\nSeq {i}: length={len(seq)}, EOS at pos {first_eos} (end of sequence)")
        else:
            print(f"\nSeq {i}: length={len(seq)}, no EOS found (reached max length)")

        decoded_text = sampler.tokenizer.decode(seq, skip_special_tokens=False)
        print(f"Generated text: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
        print("-" * 80)

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()

    print("\n✓ Test complete!")


def main_prompt():
    examples = [
        "The capital of France is",
        "The largest planet in the solar system is",
    ]

    config = Config(
        disable_sys_args=True,
        model="ar",
        batch_size=4,
        n_groups=2,
        group_size=2,
        method="greedy_map",
        transversal=True,
        gen_length=50,  # Shorter length for prompt test
    )

    print("Initializing AutoregressiveSampler for prompt-conditioned generation...")
    sampler = AutoregressiveSampler(config)

    print(f"Model max length: {sampler.model_length}")
    print(f"BOS token: {sampler.tokenizer.bos_token_id}")
    print(f"EOS token: {sampler.tokenizer.eos_token_id}")
    print(f"PAD token: {sampler.tokenizer.pad_token_id}")

    for prompt in examples:
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print("=" * 80)

        sequences, prompt_len = sampler.sample(prompt=prompt)

        print(f"\nGenerated {sequences.shape[0]} sequences")
        print(f"Prompt length: {prompt_len} tokens")

        for i, seq in enumerate(sequences):
            full_text = sampler.tokenizer.decode(seq.tolist(), skip_special_tokens=False)
            print(f"\n--- Full sequence {i} (with special tokens) ---")
            print(f"{full_text[:200]}{'...' if len(full_text) > 200 else ''}")

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()

    print("\n✓ Prompt test complete!")


if __name__ == "__main__":
    main()
    main_prompt()

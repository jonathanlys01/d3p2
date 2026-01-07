"""
Autoregressive sampler with beam-style exploration.
Mimics the behavior of diffusion samplers but uses standard left-to-right generation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

from config import Cache, Config
from subsample import get_subsample_selector
from utils import get_tokenizer, process_model_args, sample_categorical


NEG_INFINITY = -1_000_000.0
torch.set_float32_matmul_precision("high")


class AutoregressiveSampler(nn.Module):
    """Autoregressive sampler with beam-style exploration."""

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args("gpt2", cache_dir=config.cache_dir)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config
        self.tokenizer = get_tokenizer(config, "mdlm")

        # Resize model embeddings to match tokenizer vocab size (includes added PAD token)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model_length = config.gen_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    @torch.no_grad()
    def sample(self):
        batch_size = self.config.batch_size
        seq = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.int64, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        past_key_values = None

        for i in range(self.model_length):
            subsample_step = self.config.subsample_start <= i <= self.config.subsample_end
            input_ids = seq if past_key_values is None else seq[:, -1:]

            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            embeddings = outputs.hidden_states[-1] if outputs.hidden_states else None

            logits = outputs.logits[:, -1]
            probs = F.softmax(logits, dim=-1).unsqueeze(1)

            if subsample_step and self.config.group_size > 1:
                cache = Cache(
                    log_p_x0=logits.unsqueeze(1),
                    embeddings=embeddings[:, -1:] if embeddings is not None else None,
                    x=seq,
                )
                slice_idx = self.selector.subsample(cache)

                if slice_idx is not None:
                    probs = probs[slice_idx]
                    seq = seq[slice_idx]
                    finished = finished[slice_idx]

                    if past_key_values is not None:
                        past_key_values = tuple(
                            tuple(kv[slice_idx] for kv in layer_past) for layer_past in past_key_values
                        )

                    next_token = sample_categorical(probs, expand=self.config.group_size)
                    seq = seq.repeat_interleave(self.config.group_size, dim=0)
                    finished = finished.repeat_interleave(self.config.group_size, dim=0)

                    if past_key_values is not None:
                        past_key_values = tuple(
                            tuple(kv.repeat_interleave(self.config.group_size, dim=0) for kv in layer_past)
                            for layer_past in past_key_values
                        )
                else:
                    next_token = sample_categorical(probs, expand=None)
            else:
                next_token = sample_categorical(probs, expand=None)

            next_token[finished] = self.tokenizer.pad_token_id
            finished = finished | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            seq = torch.cat([seq, next_token], dim=1)

            if finished.all():
                break

        return seq


if __name__ == "__main__":
    from config import Config

    # Create a minimal config for testing
    config = Config(
        disable_sys_args=True,
        model="mdlm",
        batch_size=4,
        n_groups=2,
        group_size=2,
        method="random",
        transversal=True,
    )

    print("Initializing AutoregressiveSampler...")
    sampler = AutoregressiveSampler(config)

    print(f"Model max length: {sampler.model_length}")
    print(f"BOS token: {sampler.tokenizer.bos_token_id}")
    print(f"EOS token: {sampler.tokenizer.eos_token_id}")
    print(f"PAD token: {sampler.tokenizer.pad_token_id}")

    print("\nGenerating samples...")
    sequences = sampler.sample()

    print(f"\nGenerated {sequences.shape[0]} sequences")
    print(f"Sequence shape: {sequences.shape}")

    # Verify and display each sequence
    print("\n" + "=" * 80)
    for i, seq in enumerate(sequences):
        # Find first EOS token
        eos_positions = (seq == sampler.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            # Check that all tokens after EOS are padding
            after_eos = seq[first_eos + 1 :]
            if len(after_eos) > 0:
                all_padding = (after_eos == sampler.tokenizer.pad_token_id).all()
                print(f"\nSeq {i}: length={len(seq)}, EOS at pos {first_eos}, padding after EOS: {all_padding}")
            else:
                print(f"\nSeq {i}: length={len(seq)}, EOS at pos {first_eos} (end of sequence)")
        else:
            print(f"\nSeq {i}: length={len(seq)}, no EOS found (reached max length)")

        # Decode and display the generated text
        decoded_text = sampler.tokenizer.decode(seq, skip_special_tokens=False)
        print(f"Generated text: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
        print("-" * 80)

    print("\n✓ Test complete!")

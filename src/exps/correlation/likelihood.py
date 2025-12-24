import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from config import Config
from eval_core import Perplexity
from mdlm_ref.modeling_mdlm import MDLM
from utils import get_tokenizer, seed_all, tqdm


@torch.no_grad()
def forward_process(batch, mask_id):
    """
    Randomly mask tokens in the batch using a range of mask ratios.
    Based on the LLADA reference implementation.
    """
    b, L = batch.shape
    device = batch.device

    # Sample a starting number of tokens to mask
    k = torch.randint(1, L + 1, (), device=device)

    # Create a range of masking ratios across the batch for better MC coverage
    # This matches the logic from the LLADA reference _get_log_likelihood.py
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (L / b), steps=b, device=device)).long()
    x = ((x - 1) % L) + 1

    indices = torch.arange(L, device=device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(L)]

    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x / L).unsqueeze(1).repeat(1, L)
    return noisy_batch, p_mask, is_mask


@torch.no_grad()
def get_mdlm_log_likelihood(model, sequence, mc_num=128, batch_size=16, mask_id=None) -> float:
    """
    Estimate the log-likelihood of a sequence using Monte Carlo samples.
    Adaptation of the LLADA likelihood calculation logic for Discrete Diffusion.
    """
    device = model.device
    # Repeat sequence for batch processing of MC samples
    seq_batch = sequence[None, :].repeat(batch_size, 1).to(device)

    likelihood_sum = 0.0
    num_batches = mc_num // batch_size

    for _ in range(num_batches):
        perturbed_seq, p_mask, mask_index = forward_process(seq_batch, mask_id)

        # Use the mask ratio as the timestep for MDLM
        timesteps = p_mask[:, 0]

        output = model(perturbed_seq, timesteps=timesteps, return_dict=True)
        logits = output.logits

        # Compute Log-Likelihood estimate: log P(x) ~= E [ -CE(logits, target) / ratio ]
        # We only consider loss on masked tokens as per reference
        flat_logits = logits[mask_index]
        flat_targets = seq_batch[mask_index]
        flat_ratios = p_mask[mask_index]

        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        weighted_log_p = -ce_loss / flat_ratios

        # Aggregate log-probs per sample in the batch
        sample_indices = torch.where(mask_index)[0]
        sample_log_probs = torch.zeros(batch_size, device=device)
        sample_log_probs.index_add_(0, sample_indices, weighted_log_p)

        likelihood_sum += sample_log_probs.mean().item()

    return likelihood_sum / num_batches


def main():  # noqa: PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting MDLM Likelihood Correlation Experiment on {device}")

    seed_all(config.seed)

    # 1. Load MDLM
    print(f"Loading MDLM from {config.mdlm_model_path}...")
    mdlm_model = (
        MDLM.from_pretrained(
            config.mdlm_model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    mdlm_tokenizer = get_tokenizer(config, "mdlm")
    mask_id = mdlm_model.config.vocab_size - 1

    # 2. Load GPT-2 for Reference Quality (Perplexity)
    print("Loading GPT-2 reference model...")
    gpt2_model = AutoModel.from_pretrained("gpt2", cache_dir=config.cache_dir).to(device).eval()
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=config.cache_dir)
    ppl_evaluator = Perplexity(gpt2_model, gpt2_tokenizer)

    # 3. Data Setup
    data_path = config.data_path
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Using dummy data for demonstration if needed.")
        # Attempt to find it or exit
        if "path_to.bin" in data_path:
            print("Please provide a valid data_path in Config or via CLI.")
            return

    print(f"Loading sequences from {data_path}...")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Experiment parameters
    N_SAMPLES = 10_000
    SEQ_LENGTH = 1024
    MC_SAMPLES = 128
    MC_BATCH_SIZE = 64

    mdlm_likelihoods: list[float] = []
    gpt2_perplexities: list[float] = []

    print(f"Processing {N_SAMPLES} samples...")
    for i in tqdm(range(N_SAMPLES), desc="Likelihood Estimation"):
        start_idx = np.random.randint(0, len(data) - SEQ_LENGTH - 1)
        sample_ids = data[start_idx : start_idx + SEQ_LENGTH]

        # 1. GPT-2 Perplexity
        text = mdlm_tokenizer.decode(sample_ids, skip_special_tokens=True)
        ppl = ppl_evaluator._forward([text])[0]

        # 2. MDLM Log-Likelihood
        seq_tensor = torch.from_numpy(sample_ids.astype(np.int64)).to(device)

        ll = get_mdlm_log_likelihood(
            mdlm_model,
            seq_tensor,
            mc_num=MC_SAMPLES,
            batch_size=MC_BATCH_SIZE,
            mask_id=mask_id,
        )

        mdlm_likelihoods.append(ll)
        gpt2_perplexities.append(ppl)

    if len(mdlm_likelihoods) < 2:
        print("Insufficient data collected. Check errors above.")
        return

    mdlm_likelihoods = np.array(mdlm_likelihoods)  # type: ignore
    gpt2_perplexities = np.array(gpt2_perplexities)  # type: ignore

    # Correlation: MDLM Log-Likelihood vs Reference Log-Likelihood (approx -log PPL)
    ref_log_likelihoods = -np.log(gpt2_perplexities)

    corr, p_val = spearmanr(mdlm_likelihoods, ref_log_likelihoods)

    print("\n--- Experiment Results ---")
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"p-value: {p_val:.6e}")
    print(f"Samples processed: {len(mdlm_likelihoods)}")

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(ref_log_likelihoods, mdlm_likelihoods, alpha=0.6, c="darkblue", edgecolors="white")
    plt.xlabel("GPT-2 Log-Likelihood (Approximated as -log PPL)")
    plt.ylabel("MDLM Estimated Log-Likelihood")
    plt.title(f"Internal vs External Quality Correlation\nSpearman: {corr:.4f} (p={p_val:.2e})")
    plt.grid(True, linestyle="--", alpha=0.7)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "likelihood_correlation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

    # Save results
    np.savez(
        os.path.join(results_dir, "likelihood_results.npz"),
        mdlm_ll=mdlm_likelihoods,
        gpt2_ppl=gpt2_perplexities,
        ref_ll=ref_log_likelihoods,
    )
    print(f"Raw results saved to {results_dir}/likelihood_results.npz")


if __name__ == "__main__":
    main()

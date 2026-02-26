import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

from d5p4.mdlm_ref.modeling_mdlm import MDLM


def compute_cosine(flat: torch.Tensor) -> torch.Tensor:
    normalized_flat = F.normalize(flat, dim=-1, eps=1e-12)
    return torch.matmul(normalized_flat, normalized_flat.T)


def main():  # noqa: PLR0915
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MDLM model on {device}...")

    model = MDLM.from_pretrained("/Brain/public/models/kuleshov-group/mdlm-owt", trust_remote_code=True).to(device)

    model.eval()

    # Create dummy input based on typical sequence length
    SEQ_LEN = 128
    B = 2
    # The actual vocab size of MDLM (Llama tokenizer) is approx 32000 or 128256
    V = model.config.vocab_size

    # Generate some pseudo-random but valid looking tokens
    input_ids = torch.randint(0, V, (B, SEQ_LEN), device=device)

    print(f"\nRunning inference on dummy tokens: shape {input_ids.shape}")
    with torch.no_grad():
        # MDLM usually returns logits or a custom output object
        outputs: MaskedLMOutput = model(input_ids, return_dict=True, output_hidden_states=True)

    assert outputs.hidden_states is not None
    assert outputs.logits is not None

    logits = outputs.logits  # [B, L, V]
    embeddings = outputs.hidden_states[-1]

    print("\n--- Shapes ---")
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    _, L, H = embeddings.shape

    print("\n--- Embeddings Statistics ---")
    emb_flat = embeddings.reshape(-1, H).float()
    print(f"Mean: {emb_flat.mean().item():.4f}")
    print(f"Std: {emb_flat.std().item():.4f}")
    print(f"Min: {emb_flat.min().item():.4f}, Max: {emb_flat.max().item():.4f}")

    # Cosine similarities
    cos_sim = compute_cosine(emb_flat)
    mask = ~torch.eye(B * L, dtype=torch.bool, device=device)
    off_diag_cos = cos_sim[mask]
    print(f"Cosine Sim (off-diagonal) Mean: {off_diag_cos.mean().item():.4f}")
    print(f"Cosine Sim (off-diagonal) Std: {off_diag_cos.std().item():.4f}")

    print("\n--- Logits & Quality Statistics ---")
    log_p_x0 = F.log_softmax(logits, dim=-1).float()
    p = torch.exp(log_p_x0)

    entropy = -torch.sum(p * log_p_x0, dim=-1).view(-1)  # Flattened over B*L
    print(f"Entropy Mean: {entropy.mean().item():.4f} (Max possible: {np.log(V):.4f})")
    print(f"Entropy Std: {entropy.std().item():.4f}")
    print(f"Entropy Min: {entropy.min().item():.4f}, Max: {entropy.max().item():.4f}")

    # Quality Scores calculations
    neg_H = -entropy
    scores = (neg_H - neg_H.min()) / (neg_H.max() - neg_H.min() + 1e-12)
    print(f"Quality Score (0-1) Mean: {scores.mean().item():.4f}")
    print(f"Quality Score (0-1) Std: {scores.std().item():.4f}")

    sorted_logits, _ = torch.sort(logits.view(-1, V), dim=-1, descending=True)
    mean_sorted_logits = sorted_logits.mean(dim=0)

    print("\nMean top-K logits differences (to see sharpness):")
    print(f"Top 1: {mean_sorted_logits[0].item():.2f}")
    print(f"Top 2: {mean_sorted_logits[1].item():.2f}")
    print(f"Top 10: {mean_sorted_logits[9].item():.2f}")
    print(f"Top 50: {mean_sorted_logits[49].item():.2f}")
    print(f"Top 1000: {mean_sorted_logits[999].item():.2f}")
    print(f"Bottom 1: {mean_sorted_logits[-1].item():.2f}")

    print("\nNext step: update src/d5p4/subsample/test_absolute.py make_synthetic_cache to match these stats closely.")


if __name__ == "__main__":
    main()

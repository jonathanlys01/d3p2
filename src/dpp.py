from typing import Union

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP
from transformers.modeling_outputs import MaskedLMOutput

from utils import seed_all


def _cosine_kernel(embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Compute the cosine kernel from the embeddings.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    assert embeddings.ndim == 2, embeddings.shape
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    # Compute the cosine similarity
    kernel = np.dot(embeddings, embeddings.T)  # B x B

    # Remove the diagonal
    np.fill_diagonal(kernel, 0)

    return kernel


def compute_kernel(output: MaskedLMOutput, alpha: float) -> np.ndarray:
    """
    Compute the kernel matrix from the model output.
    """

    logits = output.logits  # B x T x V
    embeddings = output.hidden_states[-1]  # B x T x E
    seq_embeddings = embeddings.mean(dim=1)  # B x E

    # Compute the diagonal term
    p_x0 = torch.softmax(logits, dim=-1)
    p_x0 = torch.max(p_x0, dim=-1).values  # B x T
    p_x0 = torch.prod(p_x0, dim=-1)  # B
    p_x0 = p_x0.cpu().numpy()
    D = np.diag(p_x0)  # B x B

    # Compute the cosine kernel
    L = _cosine_kernel(seq_embeddings)  # B x B

    K = D + alpha * L  # B x B

    return K


def sample_dpp(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k rows from the embeddings using DPP.
    """

    L = _cosine_kernel(embeddings)
    dpp = FiniteDPP("likelihood", L=L)
    return dpp.sample_exact_k_dpp(size=k)


def main():
    B, E = 10, 256
    seed_all(0)

    x = np.random.rand(B, E).astype(np.float32)
    k = 5  # Number of samples to keep

    samples = sample_dpp(x, k)

    print("Samples:", samples)


if __name__ == "__main__":
    main()

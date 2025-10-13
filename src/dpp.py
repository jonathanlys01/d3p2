import numpy as np
from dppy.finite_dpps import FiniteDPP

from utils import seed_all


def sample_dpp(L: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k rows from the embeddings using DPP.
    """
    dpp = FiniteDPP("likelihood", L=L)
    return dpp.sample_exact_k_dpp(size=k)


def main():
    B, E = 100, 256
    seed_all(0)

    x = np.random.rand(B, E).astype(np.float32)
    kernel = x @ x.T  # [B, B]

    samples = sample_dpp(kernel, 2)

    print("Samples:", samples)


if __name__ == "__main__":
    main()

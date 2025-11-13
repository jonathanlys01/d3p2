from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP


# embeddings = torch.load("dpp_embeddings.pt", map_location=torch.device('cpu'))
# scores = torch.load("dpp_scores.pt", map_location=torch.device('cpu'))

k = 3
n = 4
B = k * n

scores = torch.rand(B)
embeddings = torch.randn(B, 1024, 768)
flat = embeddings.view(embeddings.size(0), -1)
flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

bias = torch.kron(torch.eye(k), torch.ones(n, n))

# -------------------- DPP Kernel Construction -------------------------
w = 1.0
w_bias = 2.0
pow = 2.0

K = flat @ flat.T
# K = torch.where(bias > 0, bias, K)
K = K * w
K += bias * w_bias
K += torch.diag(scores)


K_np = K.numpy()
eigenvalues, eigenvectors = np.linalg.eigh(K_np)


eigenvalues_modded = np.power(eigenvalues, pow).clip(min=1e-3, max=None)
K_np = eigenvectors @ np.diag(eigenvalues_modded) @ eigenvectors.T
K_np = (K_np + K_np.T) / 2  # ensure symmetry

print("Eigenvalues of K:", eigenvalues)
print("Modified Eigenvalues of K:", np.linalg.eigvalsh(K_np))


dpp = FiniteDPP("likelihood", L=K_np)
dpp.plot_kernel()


def block_argmax(scores, k):
    selected_indices = []
    block_size = scores.size(0) // k
    for i in range(k):
        block_scores = scores[i * block_size : (i + 1) * block_size]
        max_index = torch.argmax(block_scores).item() + i * block_size
        selected_indices.append(max_index)
    return selected_indices


selected = block_argmax(scores, k=k)
print("Block argmax selected indices:", selected)


def condition(indices):
    groups = {i // n for i in indices}
    return len(groups) == len(indices)


# -------------------- DPP Sampling and Analysis -------------------------

# make 1k samples and plot histogram of subset frequencies
subset_counts = Counter()
num_samples = 100
correct = 0
for _ in range(num_samples):
    sample = tuple(sorted(dpp.sample_exact_k_dpp(size=k)))
    subset_counts[sample] += 1
    correct += condition(sample)

print(f"Proportion of samples satisfying condition: {correct / num_samples:.2f}")

most_common_subsets = subset_counts.most_common(30)
subsets, counts = zip(*most_common_subsets)

print(tuple(sorted(selected)) in subset_counts)


colors = ["red" if condition(subset) else "blue" for subset in subsets]
plt.bar(range(len(subsets)), np.array(counts), color=colors)
plt.xticks(range(len(subsets)), [str(subset) for subset in subsets], rotation=90)
plt.xlabel("Subset")
plt.ylabel("Selection Frequency")
plt.title("Top 10 Subset Selection Frequencies over 1000 DPP Samples")
plt.savefig("dpp_partition_sampling.png", bbox_inches="tight")

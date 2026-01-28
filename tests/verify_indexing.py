import torch


def verify():
    B, L, V = 2, 5, 10
    group_size = 2

    # Mock data
    log_p_x0 = torch.randn(B, L, V)
    slice_idx = torch.tensor([0, 1])  # Select all parents
    expanded_idx = slice_idx.repeat_interleave(group_size)  # [0, 0, 1, 1]

    # Mock x0 (sampled tokens)
    x0_expanded_shape = (B * group_size, L)
    x0 = torch.randint(0, V, x0_expanded_shape)

    # Method 1: Expand then gather (Current heavy approach)
    log_p_x0_expanded = log_p_x0[expanded_idx]
    # gather along dim 2
    val1 = log_p_x0_expanded.gather(2, x0.unsqueeze(2)).squeeze(2)

    # Method 2: Advanced Indexing (Proposed optimization)
    # dimension 0: expanded_idx gives the parent index for each row in x0
    # dimension 1: we want all columns, so we can broadcast
    # dimension 2: the token indices in x0

    # We need to broadcast expanded_idx to match x0's shape or use it to index dim 0 directly
    # x0 has shape (B*G, L).
    # We want result of shape (B*G, L).
    # val[i, j] = log_p_x0[expanded_idx[i], j, x0[i, j]]

    # Create grid for dim 1
    seq_len_indices = torch.arange(L).unsqueeze(0).expand(B * group_size, L)

    # Expand parent indices for dim 0 to match (B*G, L) used in direct indexing?
    # Actually advanced indexing:
    # if we index with (Ind0, Ind1, Ind2) where all are same shape or broadcastable

    parent_indices = expanded_idx.unsqueeze(1).expand_as(x0)  # (B*G, L)

    val2 = log_p_x0[parent_indices, seq_len_indices, x0]

    print(f"Shape match: {val1.shape} vs {val2.shape}")
    print(f"Values match: {torch.allclose(val1, val2)}")

    assert torch.allclose(val1, val2)
    print("Verification Successful!")


if __name__ == "__main__":
    verify()

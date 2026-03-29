import torch
from d5p4.exps.correlation.common import compute_cosine_similarity_stats

def test_cosine_similarity_stats():
    # Batch size 3, Dimension 4
    # Create distinct vectors to have non-zero std
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.7071, 0.0, 0.0],
    ])
    
    # Pairwise similarities:
    # (0,1): 0.0
    # (0,2): 0.7071
    # (1,0): 0.0
    # (1,2): 0.7071
    # (2,0): 0.7071
    # (2,1): 0.7071
    
    # Total off-diagonal elements: 6
    # 0.0, 0.7071, 0.0, 0.7071, 0.7071, 0.7071
    
    stats = compute_cosine_similarity_stats(embeddings)
    
    expected_sims = torch.tensor([0.0, 0.7071, 0.0, 0.7071, 0.7071, 0.7071])
    expected_mean = expected_sims.mean().item()
    expected_std = expected_sims.std().item()
    
    assert abs(stats["mean"] - expected_mean) < 1e-4
    assert abs(stats["std"] - expected_std) < 1e-4
    print(f"Stats: {stats}")
    print("Test passed!")

if __name__ == "__main__":
    test_cosine_similarity_stats()

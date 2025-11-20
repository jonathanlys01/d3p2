import torch
import torch.nn.functional as F

from config import Cache
from subsample.base import BaseSelector


class GreedyBeamSearch(BaseSelector):
    """
    Greedy Beam Search Selector, greedy quality-only.
    Equivalent to a beam sampling if temperature is 0.
    If temperature is > 0, it is equivalent to a stochastic beam search.
    """

    def _transversal(self, cache: Cache):
        """Transversal selection: block argmax for temperature 0, block sampling for temperature > 0."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        gr_size = self.config.group_size
        n_groups = self.config.n_groups
        scores = scores.view(n_groups, gr_size)
        if self.config._temperature == 0:  # argmax for temperature 0
            indices = torch.argmax(scores, dim=1) + torch.arange(n_groups, device=self.device) * gr_size
        else:
            scaled_logits = scores / self.config._temperature
            # max trick for numerical stability
            max_logit = torch.max(scaled_logits, dim=1, keepdim=True).values
            scaled_logits = scaled_logits - max_logit
            probs = torch.exp(scaled_logits)
            probs = probs / probs.sum(dim=1, keepdim=True)
            indices = (
                torch.multinomial(probs, num_samples=1).squeeze(-1)
                + torch.arange(n_groups, device=self.device) * gr_size
            )
        return indices.to(self.device)

    def _non_transversal(self, cache: Cache):
        if (scores := self.compute_scores(cache)) is None:
            return None
        if self.config._temperature == 0:  # argmax for temperature 0
            topk = torch.topk(scores, k=self.config.n_groups).indices
        else:
            scaled_logits = scores / self.config._temperature
            # max trick for numerical stability
            max_logit = torch.max(scaled_logits)
            scaled_logits = scaled_logits - max_logit
            probs = torch.exp(scaled_logits)
            probs = probs / probs.sum()
            topk = torch.multinomial(probs, num_samples=self.config.n_groups, replacement=False)
        return topk


class _DiverseBeamSearch(BaseSelector):
    def _transversal(self, cache: Cache):
        flat = cache.embeddings.float().reshape(cache.embeddings.size(0), -1)
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

        if (scores := self.compute_scores(cache)) is None:
            return None

        if self.distributed_utils:
            flat = self.distributed_utils.all_gather_embeddings(flat)
            if flat is None:
                return None

        group_size = self.config.group_size
        n_groups = self.config.n_groups * self.distributed_mul
        alpha = self.config._diversity_alpha

        selected = torch.zeros(n_groups, dtype=torch.long, device=self.device)

        # Initialize first group
        selected[0] = torch.argmax(scores[:group_size])

        for i in range(1, n_groups):
            start_idx = group_size * i
            end_idx = group_size * (i + 1)

            candidate_embeddings = flat[start_idx:end_idx]  # [group_size, L*E]
            candidate_scores = scores[start_idx:end_idx]  # [group_size]

            prev_embeddings = flat[selected[:i]]  # [i, L*E]
            cos_sim = torch.matmul(candidate_embeddings, prev_embeddings.T)  # [group_size, i]

            adjusted_scores = candidate_scores - alpha * cos_sim.mean(dim=1)
            local_best_idx = torch.argmax(adjusted_scores)
            selected[i] = start_idx + local_best_idx

        return selected

    def _non_transversal(self, cache: Cache):
        """Non-transversal diverse beam search (Global MMR)."""
        flat = cache.embeddings.float().reshape(cache.embeddings.size(0), -1)
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

        if (scores := self.compute_scores(cache)) is None:
            return None

        if self.distributed_utils:
            flat = self.distributed_utils.all_gather_embeddings(flat)
            if flat is None:
                return None

        n_select = self.config.n_groups * self.distributed_mul
        alpha = self.config._diversity_alpha

        selected = torch.zeros(n_select, dtype=torch.long, device=self.device)
        selection_mask = torch.zeros_like(scores)

        first_idx = torch.argmax(scores)
        selected[0] = first_idx
        selection_mask[first_idx] = -torch.inf

        for i in range(1, n_select):
            candidate_scores = scores  # [N]
            prev_embeddings = flat[selected[:i]]  # [i, L*E]
            cos_sim = torch.matmul(flat, prev_embeddings.T)
            diversity_penalty = cos_sim.mean(dim=1)  # [N]
            adjusted_scores = candidate_scores - (alpha * diversity_penalty)

            adjusted_scores = adjusted_scores + selection_mask
            best_idx = torch.argmax(adjusted_scores)
            selected[i] = best_idx

            selection_mask[best_idx] = -torch.inf

        return selected


class DiverseBeamSearch(BaseSelector):
    def _transversal(self, cache: Cache):
        flat = cache.embeddings.float().reshape(cache.embeddings.size(0), -1)
        flat = F.normalize(flat, dim=-1, eps=1e-12)

        if (scores := self.compute_scores(cache)) is None:
            return None

        if self.distributed_utils:
            flat = self.distributed_utils.all_gather_embeddings(flat)
            if flat is None:
                return None

        # Setup groups for transversal constraints
        # In transversal mode, we map items to specific groups to ensure
        # we select exactly one item per group partition.
        item_size = scores.size(0)
        total_groups = self.config.n_groups * self.distributed_mul

        # Create mapping: [0, 0, ..., 1, 1, ..., K, K]
        item_to_group_id = torch.arange(
            total_groups,
            device=scores.device,
        ).repeat_interleave(item_size // total_groups)

        return _multi_diverse_beam_search_full_explore(
            scores,
            flat,
            total_groups,
            self.config._diversity_alpha,
            item_to_group_id,
        )

    def _non_transversal(self, cache: Cache):
        flat = cache.embeddings.float().reshape(cache.embeddings.size(0), -1)
        flat = F.normalize(flat, dim=-1, eps=1e-12)

        if (scores := self.compute_scores(cache)) is None:
            return None

        if self.distributed_utils:
            flat = self.distributed_utils.all_gather_embeddings(flat)
            if flat is None:
                return None

        # In non-transversal (Global MMR), every item is its own unique "group"
        # regarding exclusion, effectively ensuring we just don't pick the same item twice.
        item_size = scores.size(0)
        item_to_group_id = torch.arange(item_size, device=scores.device)

        total_groups = self.config.n_groups * self.distributed_mul

        return _multi_diverse_beam_search_full_explore(
            scores,
            flat,
            total_groups,
            self.config._diversity_alpha,
            item_to_group_id,
        )


def _multi_diverse_beam_search_full_explore(
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_groups: int,
    alpha: float,
    item_to_group_id: torch.Tensor,
) -> torch.Tensor:
    """
    Runs `item_size` (N) parallel diverse beam search selections.

    Batch Definition:
        B = N (We run one trajectory starting with each possible item).

    Optimization:
        Instead of storing all selected history for cosine calc (B, K, D),
        we maintain a running sum of selected embeddings (B, D).
        Mean Cosine Sim = (Sum_Vectors / k) . Candidate_Vector
    """
    device, dtype = scores.device, scores.dtype
    item_size = scores.size(0)  # N
    batch_size = item_size  # B = N
    emb_dim = embeddings.size(1)

    # --- Initialization ---

    # State: Indices selected for each of the B trajectories
    selected_indices = torch.zeros((batch_size, num_groups), dtype=torch.long, device=device)

    # State: Cumulative objective (Sum of adjusted scores) to determine best trajectory
    # Note: standard MMR doesn't have a global objective, but maximizing sum(adj_score)
    # is the standard proxy for "best greedy path".
    cumulative_objective = torch.zeros(batch_size, dtype=dtype, device=device)

    # Optimization: Running sum of selected embeddings for each batch
    # Shape: (B, D)
    sum_selected_emb = torch.zeros((batch_size, emb_dim), dtype=dtype, device=device)

    # Mask to prevent re-selecting groups/items
    # Shape: (B, N) initialized to 0
    mask = torch.zeros((batch_size, item_size), dtype=dtype, device=device)

    # --- Step 0: Force Initialization ---

    # Each batch index i starts with item i
    start_items = torch.arange(batch_size, device=device)  # (N,)
    selected_indices[:, 0] = start_items

    # Update State
    cumulative_objective += scores[start_items]
    sum_selected_emb += embeddings[start_items]

    # Update Mask: Block the groups of the items we just picked
    # Get group ID for every starting item
    start_groups = item_to_group_id[start_items]  # (B,)
    # Broadcast: (B, 1) == (1, N) -> (B, N) boolean mask
    group_mask = start_groups.unsqueeze(1) == item_to_group_id.unsqueeze(0)
    mask.masked_fill_(group_mask, -torch.inf)

    # --- Step 1 to K: Greedy Search ---

    for k in range(1, num_groups):
        # 1. Compute Diversity Penalty
        # We want Mean Cosine Sim between candidates and *all* previously selected items.
        # Because Dot Product is linear: Mean(A.C, B.C) = Mean(A,B).C
        # current_mean_vectors: (B, D)
        current_mean_vectors = sum_selected_emb / k

        # diversity_penalty: (B, D) @ (D, N) -> (B, N)
        diversity_penalty = torch.matmul(current_mean_vectors, embeddings.T)

        # 2. Compute Adjusted Scores (MMR)
        # scores: (N,) -> (1, N)
        # adjusted: (B, N)
        adjusted_scores = scores.unsqueeze(0) - (alpha * diversity_penalty)

        # 3. Apply Mask (Exclude taken groups/items)
        adjusted_scores += mask

        # 4. Select Best Candidate for each batch
        next_items = torch.argmax(adjusted_scores, dim=1)  # (B,)
        selected_indices[:, k] = next_items

        # 5. Update State
        # Add the score of the chosen item to the cumulative objective
        # gather: (B, N) gather (B, 1) -> (B, 1)
        chosen_scores = torch.gather(adjusted_scores, 1, next_items.unsqueeze(1)).squeeze(1)
        cumulative_objective += chosen_scores

        sum_selected_emb += embeddings[next_items]

        # Block the new groups
        new_groups = item_to_group_id[next_items]
        new_group_mask = new_groups.unsqueeze(1) == item_to_group_id.unsqueeze(0)
        mask.masked_fill_(new_group_mask, -torch.inf)

    # --- Final Selection ---

    # Pick the trajectory with the highest cumulative adjusted score
    best_batch_idx = torch.argmax(cumulative_objective).item()

    return selected_indices[best_batch_idx, :]

import torch

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


class DiverseBeamSearch(BaseSelector):
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

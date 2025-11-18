import torch

from config import Cache
from subsample.common import BaseSubsetSelector


class MCSearch(BaseSubsetSelector):
    """Monte Carlo Search Subset Selector, greedy quality-only."""

    def _transversal(self, cache: Cache):
        """Transversal Monte Carlo Search: block argmax over groups."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        gr_size = self.config.group_size
        n_groups = self.config.n_groups
        scores = scores.view(n_groups, gr_size)
        if self.config._temperature == 0:  # argmax for temperature 0
            indices = torch.argmax(scores, dim=1) + torch.arange(n_groups) * gr_size
        else:
            scaled_logits = scores / self.config._temperature
            # max trick for numerical stability
            max_logit = torch.max(scaled_logits, dim=1, keepdim=True).values
            scaled_logits = scaled_logits - max_logit
            probs = torch.exp(scaled_logits)
            probs = probs / probs.sum(dim=1, keepdim=True)
            indices = torch.multinomial(probs, num_samples=1).squeeze(-1) + torch.arange(n_groups) * gr_size
        return indices.to(self.device)

    def _non_transversal(self, cache: Cache):
        """Non-transversal Monte Carlo Search: global argmax over all samples."""
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


# TODO: Implement Beam Search methods


class DiverseBeamSearch(BaseSubsetSelector):
    def _transversal(self, cache: Cache):
        pass

    def _non_transversal(self, cache: Cache):
        pass

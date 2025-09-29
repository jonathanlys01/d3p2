"""
super class for embedding
reuse the mdlm model or use an external embedding model (see config for details)
"""

from torch import nn

from configs.config import Config


class EmbeddingModel(nn.Module):
    """Base class for embedding models."""

    def __init__(self, config: Config, mdlm=None):
        super(EmbeddingModel).__init__()
        self.embedding_type = config.embedding_type
        if self.embedding_type == "external":
            self.embedding_model = config.embedding_model
        else:
            assert mdlm is not None, "MDLM model must be provided for cached embeddings"
            self.embedding_model = mdlm

    def forward(self, x):
        """Forward pass through the embedding model."""
        raise NotImplementedError("forward method not implemented")

    def get_embedding(self, x):
        """Get the embedding for the input data."""
        raise NotImplementedError("get_embedding method not implemented")

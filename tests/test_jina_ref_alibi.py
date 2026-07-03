import torch

from d5p4.jina_ref.configuration_bert import JinaBertConfig
from d5p4.jina_ref.modeling_bert import JinaBertEncoder


def test_jina_alibi_construction_uses_default_meta_device():
    config = JinaBertConfig(
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        position_embedding_type="alibi",
        type_vocab_size=2,
        vocab_size=16,
    )

    torch.set_default_device("meta")
    try:
        encoder = JinaBertEncoder(config)
    finally:
        torch.set_default_device("cpu")

    assert encoder.alibi.device.type == "meta"
    assert encoder.alibi.shape == torch.Size([1, 2, 4, 4])

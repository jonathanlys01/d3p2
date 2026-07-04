from typing import Any, cast

import torch
from transformers.tokenization_utils_base import BatchEncoding

from d5p4.jina_ref.configuration_bert import JinaBertConfig
from d5p4.jina_ref.modeling_bert import JinaBertEncoder, JinaBertModel


class TinyTokenizer:
    def __call__(self, sentences, padding=True, truncation=True, return_tensors="pt", **kwargs):
        del padding, truncation, return_tensors, kwargs
        batch_size = len(sentences) if isinstance(sentences, list) else 1
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 0]] * batch_size),
                "attention_mask": torch.tensor([[1, 1, 0]] * batch_size),
            },
        )


def tiny_jina_config() -> JinaBertConfig:
    return JinaBertConfig(
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        position_embedding_type="alibi",
        type_vocab_size=2,
        vocab_size=16,
    )


def test_jina_alibi_construction_uses_default_meta_device():
    config = tiny_jina_config()

    torch.set_default_device("meta")
    try:
        encoder = JinaBertEncoder(config)
    finally:
        torch.set_default_device("cpu")

    assert encoder.alibi.device.type == "meta"
    assert encoder.alibi.shape == torch.Size([1, 2, 4, 4])


def test_jina_model_forward_keeps_transformers5_compatibility():
    model = JinaBertModel(tiny_jina_config())

    outputs = model(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
    )

    assert outputs.last_hidden_state.shape == torch.Size([1, 3, 8])


def test_jina_encode_keeps_transformers5_compatibility():
    model = JinaBertModel(tiny_jina_config())
    model.emb_pooler = "mean"
    model.tokenizer = cast(Any, TinyTokenizer())

    embeddings = model.encode(["a", "b"], convert_to_tensor=True, device=torch.device("cpu"))

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == torch.Size([2, 8])

import torch

from d5p4.dream_ref import UPSTREAM_MODEL_ID, UPSTREAM_REVISION
from d5p4.dream_ref.configuration_dream import DreamConfig
from d5p4.dream_ref.modeling_dream import DreamModel


def _tiny_config() -> DreamConfig:
    return DreamConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        mask_token_id=63,
        pad_token_id=0,
    )


def test_dream_reference_provenance_is_pinned():
    assert UPSTREAM_MODEL_ID == "Dream-org/Dream-v0-Instruct-7B"
    assert UPSTREAM_REVISION == "05334cb9faaf763692dcf9d8737c642be2b2a6ae"


def test_tiny_dream_forward_returns_only_final_hidden_state():
    model = DreamModel(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 63, 63]])

    output = model(
        input_ids,
        attention_mask="full",
        return_dict=True,
        output_hidden_states=True,
        last_hidden_state_only=True,
        num_logits_to_keep=3,
    )

    assert output.logits.shape == (1, 3, 64)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 1
    assert output.hidden_states[-1].shape == (1, 4, 32)


def test_tiny_dream_save_and_reload_preserves_state_dict(tmp_path):
    model = DreamModel(_tiny_config()).eval()
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.save_pretrained(tmp_path)
    model.generation_config.save_pretrained(tmp_path)

    loaded = DreamModel.from_pretrained(tmp_path)

    assert original.keys() == loaded.state_dict().keys()
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_tiny_dream_compiled_forward():
    model = torch.compile(DreamModel(_tiny_config()).eval(), backend="eager", dynamic=True)
    output = model(
        torch.tensor([[1, 2, 63, 63]]),
        attention_mask="full",
        return_dict=True,
        num_logits_to_keep=3,
    )

    assert output.logits.shape == (1, 3, 64)

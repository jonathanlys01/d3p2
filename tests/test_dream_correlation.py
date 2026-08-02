import sys
from types import SimpleNamespace
from typing import cast

import torch

from d5p4.config import Config
from d5p4.dream_ref.modeling_dream import DreamModel
from d5p4.exps.correlation.embeddings_dream import get_dream_hidden_states


def test_dream_correlation_config_preserves_yaml_paths_and_cli_precedence(tmp_path, monkeypatch):
    config_path = tmp_path / "brain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: dream",
                "dream_model_path: /Brain/public/models/Dream",
                "dream_tokenizer: /Brain/public/tokenizers/Dream",
                "mdlm_tokenizer: /Brain/public/tokenizers/gpt2",
                "cos_model_id: /Brain/public/models/Jina",
                "data_path: /Brain/private/user/data/val.bin",
                "cache_dir: /Brain/private/user/cache",
            ],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["embeddings_dream.py", f"--config={config_path}", "data_path=/Brain/private/user/data/override.bin"],
    )

    config = Config()

    assert config.dream_model_path == "/Brain/public/models/Dream"
    assert config.dream_tokenizer == "/Brain/public/tokenizers/Dream"
    assert config.mdlm_tokenizer == "/Brain/public/tokenizers/gpt2"
    assert config.cos_model_id == "/Brain/public/models/Jina"
    assert config.data_path == "/Brain/private/user/data/override.bin"
    assert config.cache_dir == "/Brain/private/user/cache"


def test_get_dream_hidden_states_uses_full_attention_without_lm_head():
    seen = {}
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    class _BaseModel(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
            seen.update(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            return SimpleNamespace(last_hidden_state=expected)

    class _FailingLMHead(torch.nn.Module):
        def forward(self, _hidden_states):
            raise AssertionError("Dream correlation must not materialize vocabulary logits")

    fake_model = SimpleNamespace(model=_BaseModel(), lm_head=_FailingLMHead())
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    output = get_dream_hidden_states(cast(DreamModel, fake_model), input_ids)

    assert output is expected
    assert seen == {
        "input_ids": input_ids,
        "attention_mask": "full",
        "use_cache": False,
        "return_dict": True,
    }

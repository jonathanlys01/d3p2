"""MDLM model for Hugging Face."""

import math
import typing

import flash_attn
import flash_attn.layers.rotary
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from torch import nn
from transformers import modeling_outputs

from mdlm.configuration_mdlm import MDLMConfig


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(  # noqa: PLR0913
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# function overload
def modulate(x, shift, scale):  # noqa: F811
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


############################################################
#                          Layers                          #
############################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(
        *x.shape[:-1],
        dim_out,
    )


############################################################
#     Embedding Layers for Timesteps and Class Labels      #
############################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch
                  element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the
        embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


############################################################
#                        Core Model                        #
############################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens: torch.Tensor = None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(
            6,
            dim=2,
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0.0, causal=False)

        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DITBackbone(nn.Module):
    def __init__(self, config: MDLMConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            blocks.append(DDiTBlock(config.hidden_dim, config.n_heads, config.cond_dim, dropout=config.dropout))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(config.hidden_dim, config.vocab_size, config.cond_dim)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma, output_hidden_states=False):
        if not self.config.time_conditioning or sigma is None:
            # sigma = torch.zeros_like(sigma)
            B = indices.shape[0]
            sigma = torch.zeros(B, device=indices.device)

        all_hidden_states = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden_states.append(x)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c)
                if output_hidden_states:
                    all_hidden_states.append(x)
            logits = self.output_layer(x, c)
        return logits, all_hidden_states


class MDLM(transformers.PreTrainedModel):
    """HF-compatible model."""

    config_class = MDLMConfig
    base_model_prefix = "mdlm"

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.backbone = DITBackbone(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.FloatTensor = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
    ) -> typing.Union[torch.Tensor, typing.Tuple, modeling_outputs.MaskedLMOutput]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits, all_hidden_states = self.backbone(
            indices=input_ids,
            sigma=timesteps,
            output_hidden_states=output_hidden_states,
        )
        if return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                loss=None,
            )
        elif output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits

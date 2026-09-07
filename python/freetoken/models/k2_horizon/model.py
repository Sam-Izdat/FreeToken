"""K2-Horizon decoder: 48 pre-norm layers, dense MLP on 0..2, MoE+MoVA on 3..47.

Residual-stream form (qwen3_5_moe engine convention), mathematically identical
to the reference (modeling_k2_horizon.py:642-730)::

    reference:  h=norm(x); h1=attn(h); h=x+h1; r=h; h=norm(h); h2=mlp(h); out=r+h2
    stream:     r=x; h=norm(x); h1=attn(h); (h,r)=norm_add(h1,r=x)->(norm(x+h1),x+h1);
                h2=mlp(h); return (h2, x+h1)
    next layer: (h,r)=norm_add(h2, x+h1)->(norm(x+h1+h2), x+h1+h2)  == reference

The add+norm is one ``forward_add_residual`` call per sublayer (two kernels
today; a fused Triton kernel can replace it later).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import K2HorizonAttention, K2HorizonMoVAAttention
from .moe import K2HorizonDenseMLP, K2HorizonSparseMoeBlock
from .norm import K2HorizonRMSNorm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _layernorm_num_groups() -> int:
    # K2-Horizon config: layernorm_num_groups=2. Hardcoded (no ModelConfig
    # field) -- same rationale as the always-built softplus gate: both the
    # IFM base and the primitive-ai NVFP4 export agree, and a variant that
    # differs would need a config field first.
    return 2


class K2HorizonDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_moe = layer_id >= config.first_k_dense_replace
        if self._is_moe:
            self.self_attn = K2HorizonMoVAAttention(config, layer_id)
            # Bank index (global minus dense prefix) for the offload cache.
            self.mlp = K2HorizonSparseMoeBlock(
                config, layer_id=layer_id - config.first_k_dense_replace
            )
        else:
            self.self_attn = K2HorizonAttention(config, layer_id)
            self.mlp = K2HorizonDenseMLP(config)
        self.input_layernorm = K2HorizonRMSNorm(
            config.hidden_size, _layernorm_num_groups(), eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = K2HorizonRMSNorm(
            config.hidden_size, _layernorm_num_groups(), eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class K2HorizonModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [K2HorizonDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = K2HorizonRMSNorm(
            config.hidden_size, _layernorm_num_groups(), eps=config.rms_norm_eps
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class K2HorizonForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = K2HorizonModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def needs_mova_executor(self) -> bool:
        """True when any MoVA ``v_experts`` weight is host-resident (CPU).
        The engine uses this (``mova_backend=auto``) to decide whether to
        construct the ``CpuMovaExecutor``. GPU-resident v_experts need no
        executor (the attention runs the pure-GPU loop)."""
        for layer in self.model.layers.op_list:
            experts = getattr(layer.self_attn, "v_experts", None)
            if experts is None:
                continue
            for expert in experts.op_list:
                if expert.weight.device.type == "cpu":
                    return True
        return False

    def set_mova_executor(self, executor) -> None:
        """Attach a ``CpuMovaExecutor`` to every MoVA attention layer.
        Called by the engine after weight load (generic opt-in hook; None
        clears back to the GPU loop)."""
        for layer in self.model.layers.op_list:
            attn = layer.self_attn
            if hasattr(attn, "_mova_executor"):
                attn._mova_executor = executor


__all__ = ["K2HorizonForCausalLM"]
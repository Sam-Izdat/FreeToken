"""K2-Horizon routed MoE: sigmoid+bias top-k of 100 + shared expert.

Router (reference modeling_k2_horizon.py:552-609):

    logits = gate(x)                                        # Linear(h, 100, bias=True)
    scores = sigmoid(logits.float32)                        # [N, 100]
    select = scores + gate.bias                             # bias steers top-k only
    idx    = topk(select, top_k).indices
    w      = gather(scores, idx)                            # raw sigmoid at the picks
    w      = w / w.sum(-1, keepdim=True)                    # norm_topk_prob=True
    w      = w * scaling_factor                             # router_scaling_factor=2.5

The shared expert is a plain BF16 SwiGLU MLP (intermediate=768,
``num_shared_experts=1``) added to the routed output *without* any extra
gating (unlike Qwen3.5's sigmoid(shared_gate) factor -- the reference just
adds ``self.shared_experts(residuals)``).

Expert compute goes through ``make_moe_layer`` + ``routed_forward`` so the
block is backend-agnostic: offload (NVFP4 banks on this box), fused, cpu, or
hybrid all work. ``layer_id`` is the *bank* index (global minus dense prefix);
the decoder passes ``layer_id - config.first_k_dense_replace``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer, silu_and_mul

from .attention import calc_router_weights

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class K2HorizonSharedExpert(BaseOP):
    """Always-on BF16 SwiGLU expert (intermediate = moe_intermediate_size)."""

    def __init__(self, config: ModelConfig):
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        self.gate_proj = LinearReplicated(hidden, inter, has_bias=False)
        self.up_proj = LinearReplicated(hidden, inter, has_bias=False)
        self.down_proj = LinearReplicated(inter, hidden, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat(
            [self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1
        )
        return self.down_proj.forward(silu_and_mul(gate_up))


class K2HorizonDenseMLP(BaseOP):
    """Dense SwiGLU MLP for layers 0..2 (intermediate = config.intermediate_size)."""

    def __init__(self, config: ModelConfig):
        hidden, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = LinearReplicated(hidden, inter, has_bias=False)
        self.up_proj = LinearReplicated(hidden, inter, has_bias=False)
        self.down_proj = LinearReplicated(inter, hidden, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat(
            [self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1
        )
        return self.down_proj.forward(silu_and_mul(gate_up))


class K2HorizonSparseMoeBlock(BaseOP):
    """Routed MoE + shared expert for the sparse layers (3..47)."""

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        # Gate WITH bias (moe_gate_bias=True). The bias is read back at
        # forward time for the selection-score add.
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=True)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=False,  # routing weights pre-normalized by calc_router_weights
            weight_format="bf16",
        )
        self.shared_experts = K2HorizonSharedExpert(config)
        self._router_scaling = config.routed_scaling_factor
        self._top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        # Shared expert first: the fused MoE kernel may write into its input
        # in place (HF also evaluates the shared expert first).
        shared = self.shared_experts.forward(flat)
        # Raw logits WITHOUT the gate bias: HF computes F.linear(x, W) and adds
        # the bias only to the top-k SELECTION scores (routing_weights_for_choice),
        # never to the logits. LinearReplicated(fwd) would add bias here; bypass
        # it and let calc_router_weights apply the bias to selection only.
        router_logits = torch.nn.functional.linear(flat, self.gate.weight)
        topk_weights, topk_ids = calc_router_weights(
            router_logits, self.gate.bias,
            top_k=self._top_k, scaling_factor=self._router_scaling,
        )
        # Offload kernels (lru_ensure) require int32 ids; torch.topk yields
        # int64. Fresh tensor from topk -> still safe for in-place mutation.
        topk_ids = topk_ids.to(torch.int32)
        routed = self.experts.routed_forward(flat, topk_weights, topk_ids)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["K2HorizonSparseMoeBlock", "K2HorizonSharedExpert", "K2HorizonDenseMLP"]
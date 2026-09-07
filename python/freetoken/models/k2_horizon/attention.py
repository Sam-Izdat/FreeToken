"""K2-Horizon attention: standard GQA + Mixture-of-Values Attention (MoVA).

Both classes share the same q/k/o + RoPE + (optional softplus gate) backbone
and differ only in how the *value* stream is built:

* ``K2HorizonAttention`` (dense layers 0..2): single ``v_proj`` Linear.
* ``K2HorizonMoVAAttention`` (sparse layers 3..47): ``v_router`` (Linear
  hidden->64, with bias) routes each token to top-4 of 64 ``v_experts``
  (Linear hidden->kv_attn_dim, no bias), exactly like the routed MoE but with
  a silu activation applied to each selected expert's output before the
  weighted sum. The routing weights use the same sigmoid+bias+normalize+2.5x
  formula as the routed MoE (``calc_router_weights`` below) -- the two blocks
  are intentionally identical so a shared helper can't drift.

Post-attention (both classes): the optional ``gate_proj`` output gate.
``attention_gate_func`` is "softplus" in the K2-Horizon config, applied AFTER
attention and BEFORE ``o_proj``::

    gate = softplus(gate_proj(x), beta=ln 2)      # [N, num_q, head_dim]
    attn_output = attn_output * gate
    out = o_proj(attn_output)

No q/k norms (``query_key_norm=False``), no sliding window, full RoPE over
the whole 128-dim head (``rope_head_dim == head_dim`` so the simple
``apply_rotary_pos_emb(q, k, cos, sin)`` path fires).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.models.quant_linear import make_replicated

from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def calc_router_weights(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor | None,
    *,
    top_k: int,
    scaling_factor: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid+bias+normalize+scale routing shared by the routed MoE and MoVA.

    Mirrors ``calc_router_weights`` in the reference modeling_k2_horizon.py
    (lines 136-166) plus the MoE block's ``norm_topk_prob``/``router_scaling``
    application (lines 562-577):

        scores  = sigmoid(logits.float32)                 # [N, E]
        select  = scores + router_bias                    # bias affects top-k only
        idx     = topk(select, top_k).indices
        weights = gather(scores, idx)                     # raw sigmoid at the picks
        if top_k > 1:                                     # config.norm_topk_prob=True
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * scaling_factor                # config.router_scaling_factor=2.5

    Returned as ``(weights float32, indices int64)`` -- the engine-wide ``TopK``
    contract (layers/moe.py:17). Callers needing int32 ids (offload kernels)
    cast; fp32 weights are consumed verbatim. (The HF reference casts to the
    hidden dtype at the end; fp32 here is strictly more precise and matches
    what ``fused_topk`` emits on the generic path.)
    """
    routing_scores = torch.sigmoid(router_logits.to(torch.float32))
    selection_scores = routing_scores
    if router_bias is not None:
        selection_scores = selection_scores + router_bias.to(selection_scores.dtype)
    selected_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    routing_weights = torch.gather(routing_scores, dim=-1, index=selected_indices)
    if top_k > 1:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    if scaling_factor is not None:
        routing_weights = routing_weights * scaling_factor
    return routing_weights, selected_indices


class _K2HorizonAttentionBase(BaseOP):
    """Shared q/k/o + RoPE + optional softplus-gate backbone (value stream abstract)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_attn_dim = self.num_q * self.head_dim
        self.kv_attn_dim = self.num_kv * self.head_dim
        self.q_proj = make_replicated(config, config.hidden_size, self.qo_attn_dim)
        self.k_proj = make_replicated(config, config.hidden_size, self.kv_attn_dim)

        # Post-attention output gate. Both the IFM base and the primitive-ai NVFP4
        # export carry attention_gate_func="softplus", so it is always built here.
        # (A future variant without it would need a ModelConfig field first.)
        self.gate_proj = make_replicated(config, config.hidden_size, self.qo_attn_dim)

        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=(
                tuple(rotary.scaling.items()) if rotary.scaling else None
            ),
        )
        self.o_proj = make_replicated(config, self.qo_attn_dim, config.hidden_size)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (q, k, v): q [N, num_q, head_dim] post rope, k [N, num_kv*hd]
        post rope, v [N, num_kv*hd] (built by the subclass in MoVA's case)."""
        raise NotImplementedError

    def _apply_gate(self, attn_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # F.softplus beta defaults to 1; the reference uses beta=ln 2.
        gate = F.softplus(self.gate_proj.forward(x), beta=math.log(2))
        gate = gate.view(attn_out.shape[0], self.num_q, self.head_dim)
        return attn_out * gate

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = self._qkv(x)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        gated = self._apply_gate(o, x)
        return self.o_proj.forward(gated.reshape(-1, self.qo_attn_dim))


class K2HorizonAttention(_K2HorizonAttentionBase):
    """Standard GQA for the dense layers (0..2): separate q/k/v/o + RoPE."""

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.v_proj = make_replicated(config, config.hidden_size, self.kv_attn_dim)

    def _qkv(self, x: torch.Tensor):
        positions = get_global_ctx().batch.positions
        q = self.q_proj.forward(x).view(-1, self.num_q, self.head_dim)
        k = self.k_proj.forward(x).view(-1, self.num_kv, self.head_dim)
        v = self.v_proj.forward(x).contiguous()
        q, k = self.rotary.forward(positions, q.reshape(-1, self.qo_attn_dim),
                                   k.reshape(-1, self.kv_attn_dim))
        # Backend contract (qwen3_5_moe): q [N, num_q, head_dim] 3D, k/v 2D row-major.
        return (q.view(-1, self.num_q, self.head_dim),
                k.reshape(-1, self.kv_attn_dim), v)


class K2HorizonMoVAAttention(_K2HorizonAttentionBase):
    """MoVA for the sparse layers (3..47): the value stream is routed.

    Each token picks top-4 of 64 ``v_experts`` via ``v_router``; each selected
    expert projects hidden->kv_attn_dim, passes through silu, and is weighted
    by the (normalized, 2.5x-scaled) sigmoid routing weight. The expert outputs
    are summed into the per-token value vector, which then feeds the standard
    GQA attention as V (and is written to the KV cache as V like any other
    value tensor).
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.num_value_experts = 64
        self.num_value_experts_per_tok = 4
        # v_router: hidden -> 64 with bias. The bias ONLY steers the top-k
        # selection (never the routing weights themselves); see calc_router_weights.
        self.v_router = LinearReplicated(config.hidden_size, self.num_value_experts, has_bias=True)
        # v_experts: 64 x (hidden -> kv_attn_dim), no bias, BF16 (in the ignore
        # list; never quantized -- vLLM stacks expert.weight for fused_mova_impl).
        self.v_experts = OPList([LinearReplicated(config.hidden_size, self.kv_attn_dim, has_bias=False)
                                 for _ in range(self.num_value_experts)])
        self._router_scaling = config.routed_scaling_factor
        # Host-node CPU executor for CPU-resident v_experts (attached by the
        # engine via set_mova_executor). None -> pure-GPU Python loop (used
        # when v_experts fit VRAM, and by tests driving the module directly).
        self._mova_executor = None

    def _route_values(self, x: torch.Tensor) -> torch.Tensor:
        # Router runs on GPU (v_router is 0.3 MB; always GPU-resident). The
        # value-expert dispatch follows weight placement:
        #  * GPU-resident v_experts (big-VRAM systems): pure-GPU Python loop.
        #  * CPU-resident v_experts (this box): CpuMovaExecutor host-node
        #    (graph-capturable cudaLaunchHostFunc submit/sync + pinned IO;
        #    see moe/cpu_mova_executor.py). Attached by the engine.
        in_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1])
        # Raw value-router logits WITHOUT the bias (HF: F.linear(x, W)); the bias
        # steers top-k selection only. LinearReplicated(fwd) would add it here.
        router_logits = torch.nn.functional.linear(flat, self.v_router.weight)
        routing_weights, selected = calc_router_weights(
            router_logits, self.v_router.bias, top_k=self.num_value_experts_per_tok,
            scaling_factor=self._router_scaling,
        )
        executor = self._mova_executor
        if executor is not None:
            # Host-node path: same TopK contract as the MoE executor
            # (fp32 weights, int32 ids). Returns GPU [N, kv_dim].
            v = executor.decode(self.layer_id, flat, routing_weights,
                                selected.to(torch.int32))
            return v.to(in_dtype)
        return self._route_values_gpu_loop(flat, routing_weights, selected).to(in_dtype)

    def _route_values_gpu_loop(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor, selected: torch.Tensor,
    ) -> torch.Tensor:
        # GPU-resident (or test-driven) fallback: per-expert mask -> project ->
        # silu -> weight -> index_add_, mirroring combine_routed_experts in the
        # reference. Runs on whatever device the weights live on.
        dev = self.v_experts.op_list[0].weight.device
        flat = flat.to(dev)
        routing_weights = routing_weights.to(dev)
        selected = selected.to(dev)
        out = torch.zeros(flat.shape[0], self.kv_attn_dim,
                          dtype=flat.dtype, device=dev)
        expert_mask = F.one_hot(selected, num_classes=self.num_value_experts).permute(2, 1, 0)
        hit = torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten()
        for expert_idx in hit:
            e = int(expert_idx)
            topk_pos, tok_pos = torch.where(expert_mask[e])
            expert_out = self.v_experts.op_list[e].forward(flat[tok_pos])
            expert_out = F.silu(expert_out)
            expert_out = expert_out * routing_weights[tok_pos, topk_pos, None].to(expert_out.dtype)
            out.index_add_(0, tok_pos, expert_out.to(out.dtype))
        return out.contiguous()

    def _qkv(self, x: torch.Tensor):
        positions = get_global_ctx().batch.positions
        v = self._route_values(x)
        q = self.q_proj.forward(x).view(-1, self.num_q, self.head_dim)
        k = self.k_proj.forward(x).view(-1, self.num_kv, self.head_dim)
        q, k = self.rotary.forward(positions, q.reshape(-1, self.qo_attn_dim),
                                   k.reshape(-1, self.kv_attn_dim))
        # Backend contract: q 3D, k/v 2D row-major.
        return (q.view(-1, self.num_q, self.head_dim),
                k.reshape(-1, self.kv_attn_dim), v)


__all__ = ["K2HorizonAttention", "K2HorizonMoVAAttention", "calc_router_weights"]
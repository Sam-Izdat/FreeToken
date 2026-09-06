"""Parse a K2-Horizon HF config.json into the FT ``ModelConfig`` shape.

All architectural fields used by the model code (MoE routing, MoVA, RMSNorm
group size, softplus attention gate, dense layer prefix) live in
``ModelConfig`` -- no per-model opaque payload is needed for K2-Horizon.

Quantization mapping:
  * Routed experts only are NVFP4 (per the model's `quantization_config`,
    the lone config_group targets ``mlp.experts.{N}.{gate,up,down}_proj``).
    That makes ``expert_quant="nvfp4"`` and ``dense_quant="none"`` /
    ``attn_quant="none"`` / ``lm_head_quant="none"``. The dense MLP (layers
    0..2) and shared experts stay BF16 even though they live in the routed
    expert's target regex's broader namespace -- we route by *layer*, not by
    regex, so the regex's other matches on dense/shared projections are
    deliberately ignored at dispatch time.
  * ``nvfp4_global_reciprocal`` is set by the on-disk-naming heuristic in
    models/config.py::_nvfp4_global_reciprocal (presence of ``weight_packed``
    on a routed expert → True). No per-arch branch needed here.
"""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    _nvfp4_global_reciprocal,
)


def parse_config(hf_config: Any) -> ModelConfig:
    # K2-Horizon has no multimodal wrapper config -- the HF file is the
    # text config directly.
    text = hf_config

    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    # K2-Horizon's head_dim and rope_head_dim are equal (both 128). No partial
    # rotary, no need for rotary_dim to differ from head_dim.
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )

    # K2-Horizon is text-only; a single full-attention group covers all 48
    # layers. The model.py code branches on layer_id via ``is_moe_layer`` for
    # MoE vs dense MLP, and the attention module picks MoVA vs standard GQA
    # via the same layer_id (MoVA only on sparse layers).
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=tuple(range(text.num_hidden_layers)),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )

    # Decoder sparsity: 3 dense prefix layers (mlp_only_layers=[0,1,2]),
    # 45 sparse MoE layers (top-8 of 100, sigmoid+bias+2.5x, with 1 shared
    # expert). num_hidden_layers = 48.
    mlp_only_layers = list(getattr(text, "mlp_only_layers", None) or [])
    first_k_dense_replace = max(mlp_only_layers) + 1 if mlp_only_layers else 0

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        # MoE: 100 routed experts, top-8, intermediate=768, shared expert=1.
        num_experts=int(getattr(text, "num_experts", 0) or 0),
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        # Norm-topk-prob=True in the config; the MoE block reads this and
        # renormalizes selected-expert weights before scaling.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        moe_enabled=int(getattr(text, "num_experts", 0) or 0) > 0,
        # MoVA: routed-value attention on sparse layers. The model module
        # reads mova_num_experts > 0 to swap v_proj for v_router + v_experts.
        # The decoder layer decides per-layer which attention class to build;
        # these config fields are the input to that decision.
        # (No opaque model_args payload needed; we expose them via the standard
        # ModelConfig fields below.)
        first_k_dense_replace=first_k_dense_replace,
        # K2-Horizon has exactly 1 shared expert at intermediate_size=768.
        n_shared_experts=int(getattr(text, "num_shared_experts", 0) or 0),
        # sigmoid+bias routing -- the MoE block computes sigmoid(logits),
        # adds the gate bias to the SELECTION scores only (not the routing
        # weights themselves), takes top-k, gathers the raw sigmoid scores,
        # then renormalizes and scales by 2.5.
        routed_scaling_factor=float(getattr(text, "router_scaling_factor", 1.0) or 1.0),
        # q/k RMSNorm only when query_key_norm=True (K2-Horizon: False). The
        # attention module reads this and skips building q_norm/k_norm.
        use_qk_norm=bool(getattr(text, "query_key_norm", False)),
        has_router_bias=bool(getattr(text, "moe_gate_bias", False)),
        model_type=getattr(hf_config, "model_type", "k2_horizon"),
        architectures=getattr(hf_config, "architectures", ["K2HorizonForCausalLM"]),
        vision_config=None,
        image_token_id=None,
        attention_groups=(full_group,),
        # Quantization: only the routed experts carry NVFP4. Everything else
        # (MoVA v_experts, attention projections, shared experts, dense MLP,
        # lm_head, embed_tokens, norms) is BF16. The model card's
        # quantization_config is a single NVFP4 group targeting
        # ``mlp.experts.*``; the model's own ignore list exempts the rest.
        expert_quant="nvfp4",
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        # llm-compressor NVFP4 stores the QUANT-side per-tensor scale; the
        # bank loader reciprocates 1/x at ingest when the on-disk naming
        # carries ``weight_packed`` (heuristic in models/config.py).
        nvfp4_global_reciprocal=_nvfp4_global_reciprocal(hf_config),
    )


__all__ = ["parse_config"]
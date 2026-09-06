"""K2-Horizon weight loading: BF16 dense pass + NVFP4 routed-expert banks.

Dense pass (``iter_weights``): every BF16 tensor is yielded with its HF name
unchanged -- the checkpoint keys already match the model buffers 1:1:

* ``model.embed_tokens.weight``, ``lm_head.weight``, ``model.norm.weight``
* ``model.layers.{0..2}.self_attn.{q,k,v,o,gate}_proj.weight`` (standard GQA)
* ``model.layers.{3..47}.self_attn.{q,k,o,gate}_proj.weight`` (MoVA q/k/o/gate)
* ``model.layers.{3..47}.self_attn.v_router.{weight,bias}``
* ``model.layers.{3..47}.self_attn.v_experts.{0..63}.weight`` (BF16, never quantized)
* ``model.layers.{0..2}.mlp.{gate,up,down}_proj.weight`` (dense SwiGLU, separate)
* ``model.layers.{3..47}.mlp.gate.{weight,bias}`` (MoE router with bias)
* ``model.layers.{3..47}.mlp.shared_experts.{gate,up,down}_proj.weight`` (separate)
* ``model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`` (plain RMSNorm)

No fusions: unlike Qwen3.5, K2-Horizon keeps every projection separate
(q/k/v/o unmerged, gate/up/down unmerged, shared-expert unmerged). The model
modules consume them in exactly this layout.

Skipped in the dense pass (consumed by the NVFP4 bank loader instead):
* ``model.layers.{3..47}.mlp.experts.{0..99}.{gate,up,down}_proj.{weight_packed,
  weight_scale,weight_global_scale}`` -- 13,500 projections, the only quantized
  tensors in the checkpoint.
* ``*.input_scale`` / ``*.input_global_scale`` -- W4A4 activation scales, unused
  by the FT kernels (activations stay bf16).
* ``*.weight_scale`` / ``*.weight_global_scale`` standalone -- consumed with
  their ``weight_packed`` sibling, never yielded alone.

Routed experts (``load_nvfp4_expert_sources[_parallel]``): the 13,500 NVFP4
projections flow into the shared ``nvfp4_banks`` builder with a K2-specific
source spec (K2 key pattern, bank index = global layer - 3, llm-compressor
kind_map, per-checkpoint ``global_reciprocal``). No ``setup_offload_expert_banks``
override: with ``expert_quant="nvfp4"`` the generic ``_PROVIDERS["nvfp4"]`` path
handles bank build + backend repack directly.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from tqdm import tqdm

# Routed-expert NVFP4 tensors (raw HF key). K2-Horizon has no
# ``model.language_model.`` infix -- keys start at ``model.layers.`` directly.
_NVFP4_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight_packed|"
    r"weight_scale|weight_global_scale|input_scale|input_global_scale)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer - config.first_k_dense_replace,
    desc="K2-Horizon NVFP4 experts",
)
# Quant-scale suffixes consumed with their weight, never yielded alone.
# (``weight_packed`` is matched by _NVFP4_EXPERT_RE above; the rest land here.)
_SCALE_SUFFIXES = (
    ".weight_scale", ".weight_global_scale",
    ".input_scale", ".input_global_scale",
)


def _rename(raw_name: str) -> str | None:
    """HF key -> FreeToken state-dict key, or None to skip. Identity for K2-Horizon
    (no multimodal infix, no MTP head, no vision tower in this checkpoint)."""
    return raw_name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the BF16 dense weights. Routed NVFP4 experts are excluded here
    (the offload cache loads them via ``load_nvfp4_expert_sources``); under
    the resident path (``include_moe_experts=True``) they are still excluded
    because resident NVFP4 MoE is not implemented -- K2-Horizon serves
    offload/cpu/hybrid only."""
    if get_tp_info().size > 1:
        raise NotImplementedError("k2_horizon weight loading currently supports TP=1 only")
    if not include_non_moe:
        return
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                # Routed NVFP4 experts -> offload cache, not the dense pass.
                if _NVFP4_EXPERT_RE.search(raw_name):
                    continue
                # Standalone quant scales are consumed with their weight.
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue
                name = _rename(raw_name)
                if name is None:
                    continue
                yield name, f.get_tensor(raw_name)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Parallel reader: not implemented for K2-Horizon dense weights.

    The NVFP4 routed experts (the only tensors that benefit from parallel
    read on this 36 GB checkpoint) go through ``load_nvfp4_expert_sources_parallel``
    (the bank builder's own chunked reader), not through this hook. Raising
    here lets the caller fall back to the serial dense pass.
    """
    raise NotImplementedError(
        "k2_horizon provides no iter_weights_parallel (dense weights are BF16; "
        "use load_nvfp4_expert_sources_parallel for the experts)"
    )


def _spec_for(config) -> Nvfp4ExpertSourceSpec:
    """Per-checkpoint NVFP4 source spec: K2 key pattern + bank map, with
    ``global_reciprocal`` from the parsed config (``ModelConfig.nvfp4_global_reciprocal``,
    set by the ``weight_packed`` on-disk-naming heuristic -- True for this
    llm-compressor checkpoint).

    ``kind_map`` aliases llm-compressor naming onto the modelopt canonical
    kinds the bank builder dispatches on (same map as qwen3_5_moe).
    """
    from dataclasses import replace
    return replace(
        _NVFP4_SOURCE_SPEC,
        global_reciprocal=bool(getattr(config, "nvfp4_global_reciprocal", False)),
        kind_map={
            "weight_packed": "weight",
            "weight_global_scale": "weight_scale_2",
            "input_global_scale": "input_scale",  # alias; bank builder still skips input scales
        },
    )


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, torch.Tensor]:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up
    fused on the output-row axis, down separate; weight_global_scale carried
    as the per-row global, reciprocated per the spec)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _spec_for(config),
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    """Parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _spec_for(config),
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
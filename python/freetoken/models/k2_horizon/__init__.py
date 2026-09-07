"""K2-Horizon MoVA (IFM, Apache-2.0).

48-layer MoE + Mixture-of-Values-Attention. Layers 0..2 dense SwiGLU + GQA;
layers 3..47 routed MoE (sigmoid+bias top-8 of 100, 2.5x scale) + MoVA
(sigmoid+bias top-4 of 64 value-experts). T5-style RMSNorm (n_groups=2);
softplus-gated attention output.

Quantization (primitive-ai NVFP4): routed experts only
(``weight_packed`` + ``weight_scale`` + ``weight_global_scale``,
llm-compressor). Everything else BF16. Reuses ``nvfp4_banks`` +
``qwen3_5_moe``'s llm-compressor path (``global_reciprocal`` auto-set by the
``weight_packed`` naming heuristic).
"""

from .config import parse_config
from .model import K2HorizonForCausalLM
from .weight import (
    iter_weights,
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)

__all__ = [
    "K2HorizonForCausalLM",
    "parse_config",
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]

# Key substrings kept on host (see engine._materialize_loaded_weight_state_dict):
# the 15 GB MoVA value-expert block. v_router (0.3 MB) stays GPU-resident so
# routing runs on-device; the CpuMovaExecutor takes GPU activations + routing.
CPU_WEIGHT_SUBSTRINGS = (".self_attn.v_experts.",)

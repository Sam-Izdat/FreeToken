"""MoE dispatch fallback: when a compressed-tensors NVFP4 checkpoint actually carries
routed experts on disk (e.g. an llm-compressor NVFP4 quant of a MoE model), the loader
must skip the dense-only ``_iter_weights_compressed_tensors`` branch and fall through to
the modelopt branch (which has the offload expert bank). The decision is driven by
``models.loader.has_moe_experts`` -- this test pins that helper's two layouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.loader import has_moe_experts


def _write_single_file_checkpoint(dir_path: Path, *, with_moe: bool) -> Path:
    """Single-file safetensors checkpoint (kingjones-style)."""
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.zeros(8, 16, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(
            4, 16, dtype=torch.bfloat16
        ),
    }
    if with_moe:
        # Add a single routed expert to confirm the regex matches the modelopt layout
        # (per-expert, un-fused) on disk.
        tensors["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.zeros(
            4, 8, dtype=torch.uint8
        )
        tensors["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"] = (
            torch.zeros(4, 1, dtype=torch.float8_e4m3fn)
        )
        tensors["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"] = (
            torch.tensor(0.5)
        )
    save_file(tensors, str(dir_path / "model.safetensors"))
    return dir_path


def _write_sharded_checkpoint(dir_path: Path, *, with_moe: bool) -> Path:
    """Multi-shard safetensors checkpoint with a model.safetensors.index.json
    (nvidia-style, where ft's expert bank provider reads the weight_map)."""
    shard_a: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.zeros(8, 16, dtype=torch.bfloat16),
    }
    shard_b: dict[str, torch.Tensor] = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(
            4, 16, dtype=torch.bfloat16
        ),
    }
    if with_moe:
        # Stacked / pre-fused layout (alternative to per-expert). Either layout should
        # trip the detector -- the regex matches both ``.mlp.experts.<digit>`` and
        # ``.mlp.experts.gate_up_proj`` / ``.down_proj``.
        shard_b["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(
            2, 4, 16, dtype=torch.uint8
        )
        shard_b["model.language_model.layers.0.mlp.experts.down_proj"] = torch.zeros(
            2, 16, 4, dtype=torch.uint8
        )
    save_file(shard_a, str(dir_path / "model-00001-of-00002.safetensors"))
    save_file(shard_b, str(dir_path / "model-00002-of-00002.safetensors"))
    (dir_path / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_size": 0}, "weight_map": {'
        + ", ".join(f'"{k}": "{f}"' for f, ts in (
            ("model-00001-of-00002.safetensors", shard_a),
            ("model-00002-of-00002.safetensors", shard_b),
        ) for k in ts)
        + "}}"
    )
    return dir_path


def test_dense_single_file_has_no_moe(tmp_path: Path) -> None:
    _write_single_file_checkpoint(tmp_path, with_moe=False)
    assert has_moe_experts(str(tmp_path)) is False


def test_moe_single_file_is_detected(tmp_path: Path) -> None:
    _write_single_file_checkpoint(tmp_path, with_moe=True)
    assert has_moe_experts(str(tmp_path)) is True


def test_dense_sharded_has_no_moe(tmp_path: Path) -> None:
    _write_sharded_checkpoint(tmp_path, with_moe=False)
    assert has_moe_experts(str(tmp_path)) is False


def test_moe_sharded_stacked_layout_is_detected(tmp_path: Path) -> None:
    """The stacked layout (``model.layers.N.mlp.experts.gate_up_proj`` / ``.down_proj``)
    is the alternative to per-expert un-fused. Both should trip the detector."""
    _write_sharded_checkpoint(tmp_path, with_moe=True)
    assert has_moe_experts(str(tmp_path)) is True

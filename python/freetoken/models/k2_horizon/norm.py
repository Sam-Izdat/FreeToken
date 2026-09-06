"""K2-Horizon RMSNorm -- T5-style ``x * rsqrt(mean(x^2 over n_groups slices)) * w``.

Not Gemma-style: the weight is NOT centered at (1+w). The reference
(modeling_k2_horizon.py:613-636) computes:

    h = h.reshape(*h.shape[:-1], n_groups, -1)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * rsqrt(variance + eps)
    h = h.reshape(*h.shape[:-2], -1)
    h = self.weight * h

So the variance is computed per group (each of size ``hidden_size // n_groups``),
the rsqrt normalization is applied within the group slice, then the reshape
folds the group axis back out before the per-channel weight is applied. With
``hidden_size=2560, n_groups=2`` each group is 1280-wide.

This applies to BOTH the pre-attention / pre-MLP norms in every decoder layer
AND the model-final norm. q/k norms would use the same kernel, but the
checkpoint sets ``query_key_norm=False`` so they're not built.
"""

from __future__ import annotations

import torch

from freetoken.layers import BaseOP


class K2HorizonRMSNorm(BaseOP):
    """RMSNorm with variance computed over ``n_groups`` slices of the last dim.

    Equivalent to T5LayerNorm applied group-wise. Matches
    ``K2HorizonRMSNorm`` in the reference modeling_k2_horizon.py exactly:
    float32 reduction, group-slice variance, fp-residual cast back to input
    dtype.
    """

    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        assert hidden_size % n_groups == 0, (
            f"K2HorizonRMSNorm: hidden_size={hidden_size} not divisible by n_groups={n_groups}"
        )
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h.reshape(*h.shape[:-1], self.n_groups, -1)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        h = h.reshape(*h.shape[:-2], -1)
        h = self.weight * h
        return h.to(in_dtype)

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Stream form (qwen3_5_moe engine convention): residual += x, then norm.
        # Non-fused (two kernels); a Triton fused add+group-rmsnorm can replace
        # this later without changing semantics.
        residual = residual + x
        return self.forward(residual), residual

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, n_groups={self.n_groups}"


__all__ = ["K2HorizonRMSNorm"]
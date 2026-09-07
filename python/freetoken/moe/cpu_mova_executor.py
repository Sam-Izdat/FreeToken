"""CPU-compute MoVA value-expert executor (host-node, graph-capturable).

Architecture borrows freetoken's ``CpuMoeExecutor`` (``moe/cpu_executor.py``):
pinned host IO buffers, per-(layer, batch) persistent C++ tasks, and
``cudaLaunchHostFunc`` submit/sync host nodes so decode stays inside a single
CUDA graph. Focused subset: bf16 value experts only (no quantization formats),
single GEMV + silu + router-weighted sum (no gate/up/down, no swiglu).

Placement (the upstream-flexible part): this executor is ONLY for CPU-resident
``v_experts`` (small-VRAM boxes where the 15GB block can't fit the GPU). When
``v_experts`` are GPU-resident (robust systems), the attention module runs the
same math as a pure-GPU Python loop and this executor is never constructed.
The model decides (``needs_mova_executor``); the engine orchestrates.

v_expert layout: each expert is a bf16 ``[K, H]`` row-major weight
(``Linear[out=K, in=H]``). The C++ reads them via a ``[num_layers * E]``
pointer table built here from the live model weights (no copy -- the table
holds raw addresses, and the model is kept alive on ``self`` as a GC guard).
"""

from __future__ import annotations

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor


class CpuMovaExecutor:
    """Decode-time CPU MoVA value-expert compute over the model's host weights.

    ``model`` is the ``K2HorizonForCausalLM`` (or any model exposing MoVA
    layers with ``v_experts`` as ``[E]`` lists of bf16 ``[K, H]`` linears).
    All ``v_expert`` weights MUST be CPU-resident (asserted); the router
    (``v_router``) stays on GPU and runs there.
    """

    def __init__(
        self,
        model,
        *,
        top_k: int = 4,
        num_threads: int = 0,
        max_tokens: int = 8192,
        device: torch.device,
    ) -> None:
        from freetoken.kernel import _cpu_mova
        from freetoken.moe.cpu_executor import (
            physical_core_cpus,
            resolve_threads_and_affinity,
        )

        self._model = model  # GC guard: C++ holds raw weight addresses
        self.device = device
        self.top_k = int(top_k)

        # Collect (layer_id, expert) -> weight pointer. Layers are indexed by
        # ABSOLUTE layer id (0..num_layers-1); dense (non-MoVA) layers get null
        # pointers (never called). K/H read from the first MoVA expert so
        # per-partition (TP) sizes are exact.
        layers = model.model.layers.op_list
        num_layers = len(layers)
        ptrs: list[int] = []
        H = K = E = None
        for layer_id, layer in enumerate(layers):
            attn = layer.self_attn
            experts = getattr(attn, "v_experts", None)
            if experts is None:
                ptrs.extend([0] * 64)  # dense layer: unused (placeholder E)
                continue
            # OPList of LinearReplicated, each .weight [K, H] bf16 on CPU
            ops = experts.op_list
            if E is None:
                E = len(ops)
            assert len(ops) == E, (layer_id, len(ops), E)
            for expert in ops:
                w = expert.weight
                assert w.device.type == "cpu", (
                    f"CpuMovaExecutor requires CPU-resident v_experts; "
                    f"layer {layer_id} expert is on {w.device}"
                )
                assert w.dtype == torch.bfloat16, (layer_id, w.dtype)
                assert w.is_contiguous(), (layer_id,)
                if H is None:
                    K, H = w.shape
                else:
                    assert tuple(w.shape) == (K, H), (layer_id, tuple(w.shape))
                ptrs.append(w.data_ptr())
        assert H is not None and K is not None and E is not None
        self.num_layers = num_layers
        self.num_experts = E
        self.H, self.K = H, K

        table = torch.tensor(ptrs, dtype=torch.int64)
        self._table = table  # GC guard for the pointer array itself

        nthreads, core_ids = resolve_threads_and_affinity(num_threads)
        self._ext = _cpu_mova.CpuMovaExecutor(
            num_threads=nthreads,
            num_layers=num_layers,
            num_experts=E,
            top_k=self.top_k,
            hidden_size=H,
            value_dim=K,
            max_tokens=int(max_tokens),
            experts_ptr=table.data_ptr(),
            core_ids=core_ids,
        )
        self.num_threads = nthreads
        self.isa = self._ext.isa_name()

        self._io: dict[int, dict[str, torch.Tensor]] = {}
        self._tasks: dict[tuple[int, int], int] = {}

    def _io_for(self, bs: int) -> dict[str, torch.Tensor]:
        io = self._io.get(bs)
        if io is None:
            io = {
                "x": alloc_pinned_tensor(bs, self.H, dtype=torch.bfloat16),
                "ids": alloc_pinned_tensor(bs, self.top_k, dtype=torch.int32),
                "w": alloc_pinned_tensor(bs, self.top_k, dtype=torch.float32),
                "y": alloc_pinned_tensor(bs, self.K, dtype=torch.bfloat16),
            }
            self._io[bs] = io
        return io

    def _task_for(self, layer_id: int, bs: int) -> int:
        key = (layer_id, bs)
        task = self._tasks.get(key)
        if task is None:
            io = self._io_for(bs)
            task = self._ext.create_task(
                layer_id,
                bs,
                io["x"].data_ptr(),
                io["ids"].data_ptr(),
                io["w"].data_ptr(),
                io["y"].data_ptr(),
            )
            self._tasks[key] = task
        return task

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoVA layer's value routing on the CPU. Returns GPU [bs, K].

        All ops go on the current CUDA stream so the whole thing is captured
        into the active CUDA graph (the two host nodes carry the data
        dependency on the pinned buffers, which hold this step's real routing
        on replay). ``topk_ids`` int64/int32 accepted (cast here).
        """
        bs = hidden_states.shape[0]
        io = self._io_for(bs)
        # D2H: ship this step's activations + routing to pinned host memory.
        io["x"].copy_(hidden_states, non_blocking=True)
        io["ids"].copy_(topk_ids.to(torch.int32), non_blocking=True)
        io["w"].copy_(topk_weights.to(torch.float32), non_blocking=True)
        task = self._task_for(layer_id, bs)
        out = torch.empty(bs, self.K, device=hidden_states.device,
                          dtype=hidden_states.dtype)
        stream = torch.cuda.current_stream().cuda_stream
        self._ext.submit_with_cuda_stream(stream, task)
        self._ext.sync_with_cuda_stream(stream, task)
        out.copy_(io["y"], non_blocking=True)
        return out

    def run_eager(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Eager (non-graph) path: same math, runs the pool to completion.
        Used for prefill and for testing without a CUDA stream/graph."""
        bs = hidden_states.shape[0]
        io = self._io_for(bs)
        io["x"].copy_(hidden_states.cpu())
        io["ids"].copy_(topk_ids.to(torch.int32).cpu())
        io["w"].copy_(topk_weights.to(torch.float32).cpu())
        task = self._task_for(layer_id, bs)
        self._ext.run_task(task)
        return io["y"].to(device=hidden_states.device, dtype=hidden_states.dtype)


__all__ = ["CpuMovaExecutor"]

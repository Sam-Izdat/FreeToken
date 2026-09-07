from __future__ import annotations

import importlib.util
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
        # CPU-compute MoVA value-expert executor (K2-Horizon's 15GB v_experts on
        # small-VRAM boxes). Same host-node design as _cpu_moe (cudaLaunchHostFunc
        # submit/sync so decode stays inside one CUDA graph, persistent worker
        # pool, bf16 GEMV with runtime ISA dispatch) but a focused subset: a
        # single 2560->1024 bf16 GEMV per route + silu + weighted sum, bf16 only.
        # No flag-handshake/coordinator: MoVA's CPU GEMV is ~ms-scale per layer,
        # which dwarfs the ~30-50us host-func dispatch (the handshake only pays
        # off when the CPU op is fast enough that dispatch dominates).
        CppExtension(
            name="freetoken.kernel._cpu_mova",
            sources=[
                "python/freetoken/kernel/csrc/cpu_mova/cpu_mova_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
        # --ple-backend disk row store; Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._ple_store",
                sources=[
                    "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

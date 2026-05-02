"""Local JIT build fallback for CRUMPET development."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
_CUDA = _REPO / "cuda"
_BUILD = _REPO / "build" / "torch_extensions"
_BUILD.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MAX_JOBS", os.environ.get("CRUMPET_MAX_JOBS", "1"))
os.environ.setdefault(
    "TORCH_CUDA_ARCH_LIST",
    os.environ.get("CRUMPET_CUDA_ARCH_LIST", "8.0;8.6;9.0;12.0+PTX"),
)

_CUDA_FLAGS = ["-O3", "-lineinfo", "-Xptxas=-warn-spills"]
if os.environ.get("CRUMPET_USE_FAST_MATH", "0") == "1":
    _CUDA_FLAGS.append("--use_fast_math")

_ext = load(
    name="crumpet_ops",
    sources=[
        str(_ROOT / "torch_binding.cpp"),
        str(_CUDA / "shift_partition_3d.cu"),
        str(_CUDA / "unshift_unpartition_3d.cu"),
        str(_CUDA / "attn_mask_3d.cu"),
    ],
    extra_include_paths=[str(_ROOT)],
    extra_cuda_cflags=_CUDA_FLAGS,
    extra_cflags=["-DCUDA_KERNEL"],
    build_directory=str(_BUILD),
    verbose=os.environ.get("CRUMPET_VERBOSE_BUILD", "0") == "1",
    is_python_module=False,
)

ops = torch.ops.crumpet_ops


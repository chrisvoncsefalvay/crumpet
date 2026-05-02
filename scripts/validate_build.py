#!/usr/bin/env python3
"""Validate local CRUMPET import and a minimal round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import crumpet  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.randn((1, 14, 14, 14, 8), device=device, dtype=dtype)
    windows = crumpet.fused_shift_partition_3d(x, (7, 7, 7), (3, 3, 3))
    y = crumpet.fused_unshift_unpartition_3d(windows, 1, 14, 14, 14, 8, (7, 7, 7), (3, 3, 3))
    if not torch.equal(x, y):
        raise SystemExit("round-trip failed")
    print({"device": str(device), "dtype": str(dtype), "has_cuda_ext": crumpet._HAS_CUDA_EXT})


if __name__ == "__main__":
    main()


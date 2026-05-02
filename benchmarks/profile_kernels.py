#!/usr/bin/env python3
"""NCU profiling harness for CRUMPET kernels.

Runs a single CRUMPET kernel in a tight loop with a deterministic launch
sequence so Nsight Compute (`ncu --set full`) can attach and collect
deep metrics. Prints the average wall-clock time per launch.

Usage:
    python profile_kernels.py shift   --shape 1,98,98,98,48 --window 7,7,7 --shift 3,3,3 --dtype fp16
    python profile_kernels.py unshift --shape 1,98,98,98,48 --window 7,7,7 --shift 3,3,3 --dtype fp16
    python profile_kernels.py mask    --shape 0,98,98,98,1  --window 7,7,7 --shift 3,3,3 --dtype fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))

import crumpet  # noqa: E402


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _parse_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split(","))


def _bench(fn, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("shift", "unshift", "mask"))
    p.add_argument("--shape", default="1,98,98,98,48")
    p.add_argument("--window", default="7,7,7")
    p.add_argument("--shift", default="3,3,3")
    p.add_argument("--dtype", default="fp16", choices=list(_DTYPES))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda:0")
    dtype = _DTYPES[args.dtype]
    B, D, H, W, C = _parse_tuple(args.shape)
    ws = _parse_tuple(args.window)
    ss = _parse_tuple(args.shift)
    torch.cuda.set_device(0)

    if args.kind == "shift":
        x = torch.randn((B, D, H, W, C), device=device, dtype=dtype)

        def fn():
            return crumpet.fused_shift_partition_3d(x, ws, ss)

    elif args.kind == "unshift":
        x = torch.randn((B, D, H, W, C), device=device, dtype=dtype)
        win = crumpet.fused_shift_partition_3d(x, ws, ss)

        def fn():
            return crumpet.fused_unshift_unpartition_3d(win, B, D, H, W, C, ws, ss)

    elif args.kind == "mask":
        crumpet.clear_mask_cache()
        os_disable = __import__("os").environ
        os_disable["CRUMPET_DISABLE_MASK_CACHE"] = "1"

        def fn():
            return crumpet.compute_attn_mask_3d(D, H, W, ws, ss, dtype, device)

    else:
        raise ValueError(args.kind)

    avg_ms = _bench(fn, args.warmup, args.iters)
    print(f"kind={args.kind} shape={args.shape} window={args.window} shift={args.shift} dtype={args.dtype} avg_ms={avg_ms:.6f}")


if __name__ == "__main__":
    main()

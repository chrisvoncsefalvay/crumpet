#!/usr/bin/env python3
"""Benchmark CRUMPET kernels against PyTorch references."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import crumpet  # noqa: E402
from crumpet.reference import (  # noqa: E402
    compute_attn_mask_3d_reference,
    unfused_shift_partition_3d_reference,
    unfused_unshift_unpartition_3d_reference,
    window_partition_reference,
)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _time_cuda(fn, warmup: int, iters: int) -> dict[str, float]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
        if isinstance(out, torch.Tensor):
            out.record_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        if isinstance(out, torch.Tensor):
            out.record_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(math.ceil(0.95 * len(times))) - 1)],
        "iters": iters,
    }


def _speedup(reference: dict[str, float], kernel: dict[str, float]) -> float:
    return reference["mean_ms"] / kernel["mean_ms"] if kernel["mean_ms"] else float("inf")


def _monai_style_mask_reference(D, H, W, window_size, shift_size, dtype, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d_slice in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h_slice in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w_slice in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                cnt += 1
    mask_windows = window_partition_reference(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask.to(dtype=dtype)


def _mask_case(D, H, W, window_size, shift_size, dtype, device, warmup, iters):
    crumpet.clear_mask_cache()
    ref = compute_attn_mask_3d_reference(D, H, W, window_size, shift_size, dtype, device)
    got = crumpet.compute_attn_mask_3d(D, H, W, window_size, shift_size, dtype, device)
    torch.cuda.synchronize()
    exact = torch.equal(ref, got)
    del ref, got
    torch.cuda.empty_cache()

    crumpet.clear_mask_cache()
    ref_time = _time_cuda(
        lambda: _monai_style_mask_reference(D, H, W, window_size, shift_size, dtype, device),
        warmup,
        iters,
    )
    old_disable = os.environ.get("CRUMPET_DISABLE_MASK_CACHE")
    os.environ["CRUMPET_DISABLE_MASK_CACHE"] = "1"
    crumpet.clear_mask_cache()
    try:
        kernel_time = _time_cuda(
            lambda: crumpet.compute_attn_mask_3d(D, H, W, window_size, shift_size, dtype, device),
            warmup,
            iters,
        )
    finally:
        if old_disable is None:
            os.environ.pop("CRUMPET_DISABLE_MASK_CACHE", None)
        else:
            os.environ["CRUMPET_DISABLE_MASK_CACHE"] = old_disable
    crumpet.clear_mask_cache()
    crumpet.compute_attn_mask_3d(D, H, W, window_size, shift_size, dtype, device)
    cached = _time_cuda(
        lambda: crumpet.compute_attn_mask_3d(D, H, W, window_size, shift_size, dtype, device),
        warmup,
        iters,
    )
    return {
        "D": D,
        "H": H,
        "W": W,
        "window_size": list(window_size),
        "shift_size": list(shift_size),
        "dtype": str(dtype).replace("torch.", ""),
        "exact": exact,
        "reference": ref_time,
        "reference_kind": "monai_style_python_slices_partition_and_masked_fill",
        "kernel": kernel_time,
        "cached": cached,
        "speedup": _speedup(ref_time, kernel_time),
    }


def _partition_case(D, H, W, C, window_size, shift_size, dtype, device, warmup, iters):
    x = torch.randn((1, D, H, W, C), device=device, dtype=dtype)
    ref = unfused_shift_partition_3d_reference(x, window_size, shift_size)
    got = crumpet.fused_shift_partition_3d(x, window_size, shift_size)
    restored = crumpet.fused_unshift_unpartition_3d(got, 1, D, H, W, C, window_size, shift_size)
    torch.cuda.synchronize()
    forward_exact = torch.equal(ref, got)
    roundtrip_exact = torch.equal(restored, x)
    ref_time = _time_cuda(lambda: unfused_shift_partition_3d_reference(x, window_size, shift_size), warmup, iters)
    kernel_time = _time_cuda(lambda: crumpet.fused_shift_partition_3d(x, window_size, shift_size), warmup, iters)
    ref_unshift_time = _time_cuda(
        lambda: unfused_unshift_unpartition_3d_reference(ref, 1, D, H, W, C, window_size, shift_size),
        warmup,
        iters,
    )
    kernel_unshift_time = _time_cuda(
        lambda: crumpet.fused_unshift_unpartition_3d(got, 1, D, H, W, C, window_size, shift_size),
        warmup,
        iters,
    )
    return {
        "D": D,
        "H": H,
        "W": W,
        "C": C,
        "window_size": list(window_size),
        "shift_size": list(shift_size),
        "dtype": str(dtype).replace("torch.", ""),
        "forward_exact": forward_exact,
        "roundtrip_exact": roundtrip_exact,
        "reference_shift_partition": ref_time,
        "kernel_shift_partition": kernel_time,
        "shift_partition_speedup": _speedup(ref_time, kernel_time),
        "reference_unshift_unpartition": ref_unshift_time,
        "kernel_unshift_unpartition": kernel_unshift_time,
        "unshift_unpartition_speedup": _speedup(ref_unshift_time, kernel_unshift_time),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mask-output-json", default="benchmarks/results/mask_results.json")
    parser.add_argument("--kernel-output-json", default="benchmarks/results/kernel_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_kernel.py requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)

    mask_cases = [
        _mask_case(14, 14, 14, (7, 7, 7), (3, 3, 3), dtype, device, args.warmup, args.iters),
        _mask_case(49, 49, 49, (7, 7, 7), (3, 3, 3), dtype, device, args.warmup, args.iters),
        _mask_case(98, 98, 98, (7, 7, 7), (3, 3, 3), dtype, device, max(1, args.warmup // 2), max(3, args.iters // 4)),
    ]
    partition_cases = [
        _partition_case(14, 14, 14, 48, (7, 7, 7), (3, 3, 3), dtype, device, args.warmup, args.iters),
        _partition_case(49, 49, 49, 48, (7, 7, 7), (3, 3, 3), dtype, device, args.warmup, args.iters),
        _partition_case(49, 49, 49, 96, (7, 7, 7), (0, 0, 0), dtype, device, args.warmup, args.iters),
    ]

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mask_speedups = [case["speedup"] for case in mask_cases]
    shifted_cases = [case for case in partition_cases if case["shift_size"] != [0, 0, 0]]
    large_shifted_cases = [case for case in shifted_cases if case["D"] >= 49]
    kernel_speedups = [case["shift_partition_speedup"] for case in shifted_cases]
    large_kernel_speedups = [case["shift_partition_speedup"] for case in large_shifted_cases]
    mask_out = {
        "timestamp": timestamp,
        "phase": 1,
        "benchmark": "compute_attn_mask_3d",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "cases": mask_cases,
        "min_speedup": min(mask_speedups),
        "large_case_speedup": mask_cases[-1]["speedup"],
        "halt_rule_triggered": mask_cases[-1]["speedup"] < 10.0,
    }
    kernel_out = {
        "timestamp": timestamp,
        "phase": 2,
        "benchmark": "fused_shift_partition_and_unshift_unpartition_3d",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "cases": partition_cases,
        "min_shift_partition_speedup": min(kernel_speedups),
        "large_shift_partition_speedup": min(large_kernel_speedups),
        "halt_rule_triggered": min(large_kernel_speedups) < 1.5,
    }
    for path, payload in (
        (Path(args.mask_output_json), mask_out),
        (Path(args.kernel_output_json), kernel_out),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mask_large_case_speedup": mask_out["large_case_speedup"],
                "partition_min_speedup": kernel_out["min_shift_partition_speedup"],
                "mask_halt": mask_out["halt_rule_triggered"],
                "kernel_halt": kernel_out["halt_rule_triggered"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

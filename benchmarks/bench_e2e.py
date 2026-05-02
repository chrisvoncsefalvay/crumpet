#!/usr/bin/env python3
"""Synthetic MONAI Swin UNETR end-to-end benchmark for the CRUMPET patch."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import crumpet  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _time_model(model, x, dtype, warmup, iters):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype):
                model(x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype):
                y = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(math.ceil(0.95 * len(times))) - 1)],
        "p99_ms": max(times),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "output": y.detach(),
    }


def _make_model(spatial_size):
    from monai.networks.nets import SwinUNETR

    kwargs = {
        "in_channels": 1,
        "out_channels": 2,
        "feature_size": 12,
        "depths": (1, 1, 1, 1),
        "num_heads": (3, 6, 12, 24),
        "use_checkpoint": False,
        "spatial_dims": 3,
    }
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = spatial_size
    return SwinUNETR(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-json", default="benchmarks/results/e2e_results.json")
    args = parser.parse_args()

    import monai

    if not torch.cuda.is_available():
        raise RuntimeError("bench_e2e.py requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(123)

    crumpet.unpatch_monai_swin_unetr()
    baseline = _make_model((args.spatial_size, args.spatial_size, args.spatial_size)).to(device).eval()
    patched = _make_model((args.spatial_size, args.spatial_size, args.spatial_size)).to(device).eval()
    patched.load_state_dict(baseline.state_dict())
    x = torch.randn((1, 1, args.spatial_size, args.spatial_size, args.spatial_size), device=device)

    if args.compile:
        baseline = torch.compile(baseline, fullgraph=False)

    baseline_stats = _time_model(baseline, x, dtype, args.warmup, args.iters)
    crumpet.patch_monai_swin_unetr()
    try:
        if args.compile:
            patched = torch.compile(patched, fullgraph=False)
        patched_stats = _time_model(patched, x, dtype, args.warmup, args.iters)
    finally:
        crumpet.unpatch_monai_swin_unetr()

    y0 = baseline_stats.pop("output")
    y1 = patched_stats.pop("output")
    diff = (y0.float() - y1.float()).abs()
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 3,
        "benchmark": "monai_swin_unetr_synthetic_e2e",
        "device": str(device),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "cuda_version": torch.version.cuda,
        "kernel_version": crumpet.__version__,
        "dtype": args.dtype,
        "spatial_size": [args.spatial_size, args.spatial_size, args.spatial_size],
        "compile": bool(args.compile),
        "synthetic": True,
        "baseline": baseline_stats,
        "patched": patched_stats,
        "speedup": baseline_stats["mean_ms"] / patched_stats["mean_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "passes_tolerance": bool(diff.max().item() <= (1e-2 if dtype != torch.float32 else 1e-5)),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "speedup": out["speedup"],
                "max_abs_diff": out["max_abs_diff"],
                "passes_tolerance": out["passes_tolerance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


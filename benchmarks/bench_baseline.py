#!/usr/bin/env python3
"""Phase 0 unfused shifted-window mechanics benchmark.

This benchmark measures the PyTorch/MONAI-style operations that CRUMPET plans
to fuse. It writes structured JSON and deliberately makes no CUDA extension
calls, so it can be run before any kernel implementation exists.
"""

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

from crumpet.reference import (  # noqa: E402
    compute_attn_mask_3d_reference,
    unfused_shift_partition_3d_reference,
    unfused_unshift_unpartition_3d_reference,
)


INPUT_SHAPES = {
    "btcv": (1, 1, 96, 96, 96),
    "brats": (1, 4, 128, 128, 128),
    "large_ct": (1, 1, 192, 192, 192),
}


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _stage_shapes(input_shape: tuple[int, int, int, int, int]) -> list[dict[str, object]]:
    _, _, D, H, W = input_shape
    channels = [48, 96, 192, 384]
    out = []
    for stage, C in enumerate(channels):
        sd = max(1, D // (2 ** (stage + 1)))
        sh = max(1, H // (2 ** (stage + 1)))
        sw = max(1, W // (2 ** (stage + 1)))
        base_ws = (7, 7, 7)
        window_size = tuple(min(v, 7) for v in (sd, sh, sw))
        padded = tuple(int(math.ceil(v / ws) * ws) for v, ws in zip((sd, sh, sw), window_size))
        for block in range(2):
            if block % 2 == 0:
                shift_size = (0, 0, 0)
            else:
                shift_size = tuple(ws // 2 if dim > ws else 0 for dim, ws in zip((sd, sh, sw), window_size))
            out.append(
                {
                    "stage_index": stage,
                    "block_index": block,
                    "D": sd,
                    "H": sh,
                    "W": sw,
                    "C": C,
                    "padded_D": padded[0],
                    "padded_H": padded[1],
                    "padded_W": padded[2],
                    "window_size": list(window_size),
                    "shift_size": list(shift_size),
                }
            )
    return out


def _event_time(fn, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(math.ceil(0.95 * len(times))) - 1)],
        "iters": iters,
    }


def _attention_probe(windows: torch.Tensor, heads: int) -> torch.Tensor:
    BnW, volume, C = windows.shape
    head_dim = max(1, C // heads)
    usable = heads * head_dim
    q = windows[:, :, :usable].reshape(BnW, volume, heads, head_dim).transpose(1, 2)
    k = q
    return torch.matmul(q, k.transpose(-2, -1))


def _bench_block(
    block: dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    B = 1
    D = int(block["padded_D"])
    H = int(block["padded_H"])
    W = int(block["padded_W"])
    C = int(block["C"])
    window_size = tuple(int(v) for v in block["window_size"])
    shift_size = tuple(int(v) for v in block["shift_size"])
    x = torch.randn((B, D, H, W, C), device=device, dtype=dtype)
    windows = unfused_shift_partition_3d_reference(x, window_size, shift_size)
    heads = max(1, min(24, C // 16))

    result = dict(block)
    result["dtype"] = str(dtype).replace("torch.", "")
    result["device"] = str(device)
    result["num_windows"] = int(windows.shape[0])
    result["window_volume"] = int(windows.shape[1])
    result["roll_partition"] = _event_time(
        lambda: unfused_shift_partition_3d_reference(x, window_size, shift_size),
        device,
        warmup,
        iters,
    )
    result["window_reverse_roll"] = _event_time(
        lambda: unfused_unshift_unpartition_3d_reference(
            windows, B, D, H, W, C, window_size, shift_size
        ),
        device,
        warmup,
        iters,
    )
    result["compute_mask"] = _event_time(
        lambda: compute_attn_mask_3d_reference(D, H, W, window_size, shift_size, dtype, device),
        device,
        max(1, warmup // 2),
        max(1, iters // 2),
    )
    result["attention_probe"] = _event_time(
        lambda: _attention_probe(windows, heads),
        device,
        max(1, warmup // 2),
        max(1, iters // 2),
    )

    fusable = (
        result["roll_partition"]["mean_ms"]
        + result["window_reverse_roll"]["mean_ms"]
        + result["compute_mask"]["mean_ms"]
    )
    total_probe = fusable + result["attention_probe"]["mean_ms"]
    result["fusable_over_attention_probe_fraction"] = fusable / total_probe if total_probe else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(INPUT_SHAPES), default="btcv")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    blocks = _stage_shapes(INPUT_SHAPES[args.shape])[: args.max_blocks]
    records = [_bench_block(block, dtype, device, args.warmup, args.iters) for block in blocks]
    fractions = [
        r["fusable_over_attention_probe_fraction"]
        for r in records
        if r["fusable_over_attention_probe_fraction"] is not None
    ]
    mean_fraction = statistics.fmean(fractions) if fractions else None
    if mean_fraction is None:
        decision = "unknown"
    elif mean_fraction < 0.05:
        decision = "halt_and_pivot_to_full_window_msa_fusion"
    elif mean_fraction <= 0.10:
        decision = "proceed_with_modest_expectations"
    else:
        decision = "proceed_fully"

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 0,
        "benchmark": "unfused_shifted_window_mechanics",
        "mode": "synthetic_internal_swin_unetr_shapes",
        "shape": args.shape,
        "input_shape": list(INPUT_SHAPES[args.shape]),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "mean_fusable_over_attention_probe_fraction": mean_fraction,
        "go_no_go_decision": decision,
        "notes": [
            "This benchmark does not call CRUMPET kernels.",
            "The attention probe is a window-local QK matmul timing, not a full MONAI block timing.",
            "Full MONAI module timing is deferred to bench_e2e.py.",
        ],
        "records": records,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("go_no_go_decision", "mean_fusable_over_attention_probe_fraction")}, indent=2))


if __name__ == "__main__":
    main()


"""BTCV Swin UNETR demo path for CRUMPET.

The real path expects a MONAI model-zoo Swin UNETR BTCV bundle and a user
supplied abdominal CT NIfTI. If no image is supplied, the command runs a
clearly labelled synthetic smoke test only.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

from . import __version__, patch_monai_swin_unetr, unpatch_monai_swin_unetr


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _make_model(roi_size, synthetic: bool):
    from monai.networks.nets import SwinUNETR

    kwargs = {
        "in_channels": 1,
        "out_channels": 2 if synthetic else 14,
        "feature_size": 12 if synthetic else 48,
        "use_checkpoint": False,
        "spatial_dims": 3,
    }
    if synthetic:
        kwargs["depths"] = (1, 1, 1, 1)
        kwargs["num_heads"] = (3, 6, 12, 24)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = tuple(roi_size)
    return SwinUNETR(**kwargs)


def _load_checkpoint(model, bundle_dir: Path, device: torch.device) -> dict[str, object]:
    checkpoint = bundle_dir / "models" / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"expected MONAI bundle checkpoint at {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "network", "net"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    result = model.load_state_dict(cleaned, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _load_image(path: Path) -> torch.Tensor:
    import nibabel as nib

    img = nib.load(str(path)).get_fdata(dtype="float32")
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    x = x.clamp(-175, 250)
    x = (x + 175.0) / 425.0
    return x


def _load_label(path: Path) -> torch.Tensor:
    import nibabel as nib

    label = nib.load(str(path)).get_fdata(dtype="float32")
    return torch.from_numpy(label).long().unsqueeze(0)


def _dice(pred: torch.Tensor, label: torch.Tensor, classes: int) -> float | None:
    if pred.shape != label.shape:
        return None
    scores = []
    for cls in range(1, classes):
        p = pred == cls
        l = label == cls
        denom = p.sum() + l.sum()
        if denom.item() == 0:
            continue
        scores.append((2 * (p & l).sum().float() / denom.float()).item())
    return float(sum(scores) / len(scores)) if scores else None


def _run_inference(model, x, roi_size, sw_batch_size, dtype, warmup, iters):
    from monai.inferers import sliding_window_inference

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype):
                sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=0.0)
    torch.cuda.synchronize()
    times = []
    out = None
    with torch.no_grad():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype):
                out = sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=0.0)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    times_sorted = sorted(times)
    return {
        "mean_ms": float(sum(times) / len(times)),
        "p50_ms": float(times_sorted[len(times_sorted) // 2]),
        "p95_ms": float(times_sorted[min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))],
        ),
        "p99_ms": float(times_sorted[-1]),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "output": out.detach(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="./bundles/swin_unetr_btcv_segmentation")
    parser.add_argument("--image")
    parser.add_argument("--label")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-json", default="benchmarks/results/btcv_bundle_demo.json")
    args = parser.parse_args(argv)

    import monai

    if not torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        raise RuntimeError("CUDA is required for the BTCV demo")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    synthetic = args.image is None
    bundle_dir = Path(args.bundle_dir)

    model = _make_model(args.roi_size, synthetic=synthetic).to(device).eval()
    checkpoint_info = None
    if not synthetic:
        checkpoint_info = _load_checkpoint(model, bundle_dir, device)

    if synthetic:
        x = torch.randn((1, 1, *args.roi_size), device=device)
        label = None
    else:
        x = _load_image(Path(args.image)).to(device)
        label = _load_label(Path(args.label)).to(device) if args.label else None

    # When --compile is set, the patched and baseline paths each get their
    # own freshly compiled graph. The first compile traces through the
    # eager / unpatched MONAI methods, so calling torch.compile() once
    # before flipping the patch leaves the patched run executing the
    # *unpatched* dynamo graph.
    #
    # We use mode="default" rather than "reduce-overhead" because the
    # latter wraps in CUDA graphs that fight CRUMPET's caches (the mask
    # cache and the attention's relative-position bias would be seen as
    # overwritten across iterations). The default mode still fuses
    # elementwise ops and recovers the bulk of the compile benefit.
    def _maybe_compile(m):
        if not args.compile:
            return m
        return torch.compile(m, mode="default", fullgraph=False)

    unpatch_monai_swin_unetr()
    if args.compile:
        # Reset dynamo's compile cache so the baseline trace is clean.
        torch._dynamo.reset()
    baseline_model = _maybe_compile(model)
    baseline = _run_inference(baseline_model, x, tuple(args.roi_size), args.sw_batch_size, dtype, args.warmup, args.iters)

    patch_monai_swin_unetr()
    try:
        if args.compile:
            torch._dynamo.reset()
        patched_model = _maybe_compile(model)
        patched = _run_inference(patched_model, x, tuple(args.roi_size), args.sw_batch_size, dtype, args.warmup, args.iters)
    finally:
        unpatch_monai_swin_unetr()

    y0 = baseline.pop("output")
    y1 = patched.pop("output")
    diff = (y0.float() - y1.float()).abs()
    pred0 = y0.argmax(dim=1)
    pred1 = y1.argmax(dim=1)
    baseline_dice = _dice(pred0, label, y0.shape[1]) if label is not None else None
    patched_dice = _dice(pred1, label, y1.shape[1]) if label is not None else None
    dice_delta = (
        patched_dice - baseline_dice
        if baseline_dice is not None and patched_dice is not None
        else None
    )

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "cuda_version": torch.version.cuda,
        "kernel_version": __version__,
        "bundle": "swin_unetr_btcv_segmentation",
        "bundle_dir": str(bundle_dir),
        "checkpoint": checkpoint_info,
        "image": args.image,
        "label": args.label,
        "dtype": args.dtype,
        "roi_size": list(args.roi_size),
        "sw_batch_size": args.sw_batch_size,
        "compile": bool(args.compile),
        "baseline": baseline,
        "patched": patched,
        "speedup": baseline["mean_ms"] / patched["mean_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "baseline_dice": baseline_dice,
        "patched_dice": patched_dice,
        "dice_delta": dice_delta,
        "synthetic": synthetic,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"synthetic": synthetic, "speedup": out["speedup"], "max_abs_diff": out["max_abs_diff"]}, indent=2))


if __name__ == "__main__":
    main()

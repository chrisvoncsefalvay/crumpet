"""CRUMPET: fused 3D shifted-window kernels for efficient transformers."""

from __future__ import annotations

import contextlib
import os
from typing import Sequence

import torch

from .cache import (
    clear_mask_cache,
    get_or_compute_mask,
    make_mask_key,
    mask_cache_info,
)
from .reference import (
    compute_attn_mask_3d_reference,
    unfused_shift_partition_3d_reference,
    unfused_unshift_unpartition_3d_reference,
    window_partition_reference,
    window_reverse_reference,
)

__version__ = "0.1.0"

try:
    from ._ops import ops as _ops

    _HAS_CUDA_EXT = True
except ModuleNotFoundError:
    if torch.cuda.is_available() or os.environ.get("CRUMPET_BUILD_JIT_WITHOUT_CUDA", "0") == "1":
        try:
            from ._jit_ops import ops as _ops

            _HAS_CUDA_EXT = True
        except Exception:
            _ops = None
            _HAS_CUDA_EXT = False
    else:
        _ops = None
        _HAS_CUDA_EXT = False
except Exception:
    _ops = None
    _HAS_CUDA_EXT = False

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _register_fake_impls() -> None:
    if not _HAS_CUDA_EXT:
        return
    register = getattr(torch.library, "register_fake", None)
    if register is None:
        return

    def _noop(*args, **kwargs):
        return None

    for name in (
        "compute_attn_mask_3d",
        "fused_shift_partition_3d",
        "fused_shift_partition_3d_backward",
        "fused_unshift_unpartition_3d",
        "fused_unshift_unpartition_3d_backward",
    ):
        try:
            register(f"crumpet_ops::{name}")(_noop)
        except (RuntimeError, AttributeError, ValueError):
            pass


_register_fake_impls()


@contextlib.contextmanager
def _nvtx_range(name: str):
    enabled = os.environ.get("CRUMPET_ENABLE_NVTX", "0") == "1"
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _triple(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    out = tuple(int(v) for v in value)
    if any(v < 0 for v in out):
        raise ValueError(f"{name} values must be non-negative")
    return out


def _validate_shape(
    D: int,
    H: int,
    W: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> None:
    if D <= 0 or H <= 0 or W <= 0:
        raise ValueError("D, H and W must be positive")
    if any(v <= 0 for v in window_size):
        raise ValueError("window_size values must be positive")
    if D % window_size[0] or H % window_size[1] or W % window_size[2]:
        raise ValueError(
            "standalone CRUMPET kernels require D, H and W to be divisible by window_size"
        )
    for axis, (shift, window) in enumerate(zip(shift_size, window_size)):
        if shift < 0 or shift >= window:
            raise ValueError(
                f"shift_size[{axis}] must satisfy 0 <= shift < window_size[{axis}]"
            )


def _check_dtype(dtype: torch.dtype) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError("CRUMPET supports torch.float32, torch.float16 and torch.bfloat16")


def _force_reference() -> bool:
    return os.environ.get("CRUMPET_FORCE_REFERENCE", "0") == "1"


def _can_use_cuda_tensor(x: torch.Tensor) -> bool:
    return (
        _HAS_CUDA_EXT
        and not _force_reference()
        and x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.device.type != "meta"
    )


def compute_attn_mask_3d(
    D: int,
    H: int,
    W: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Compute a shifted 3D Swin attention mask."""

    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_dtype(dtype)
    D, H, W = int(D), int(H), int(W)
    _validate_shape(D, H, W, window_size, shift_size)
    dev = torch.device(device)
    volume = window_size[0] * window_size[1] * window_size[2]
    num_windows = (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])

    if dev.type == "meta":
        return torch.empty((num_windows, volume, volume), dtype=dtype, device=dev)

    def compute() -> torch.Tensor:
        if _HAS_CUDA_EXT and not _force_reference() and dev.type == "cuda":
            out = torch.empty((num_windows, volume, volume), dtype=dtype, device=dev)
            with _nvtx_range("crumpet.compute_attn_mask_3d"):
                _ops.compute_attn_mask_3d(
                    out,
                    D,
                    H,
                    W,
                    window_size[0],
                    window_size[1],
                    window_size[2],
                    shift_size[0],
                    shift_size[1],
                    shift_size[2],
                )
            return out
        return compute_attn_mask_3d_reference(D, H, W, window_size, shift_size, dtype, dev)

    key = make_mask_key(D, H, W, window_size, shift_size, dtype, dev)
    return get_or_compute_mask(key, compute)


class _ShiftPartitionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, window_size, shift_size):
        x = x.contiguous()
        B, D, H, W, C = (int(v) for v in x.shape)
        volume = window_size[0] * window_size[1] * window_size[2]
        num_windows = (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])
        out = torch.empty((B * num_windows, volume, C), dtype=x.dtype, device=x.device)
        _ops.fused_shift_partition_3d(out, x, *window_size, *shift_size)
        ctx.shape = (B, D, H, W, C)
        ctx.window_size = window_size
        ctx.shift_size = shift_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        B, D, H, W, C = ctx.shape
        grad_x = torch.empty((B, D, H, W, C), dtype=grad_output.dtype, device=grad_output.device)
        _ops.fused_shift_partition_3d_backward(
            grad_x,
            grad_output.contiguous(),
            B,
            D,
            H,
            W,
            C,
            *ctx.window_size,
            *ctx.shift_size,
        )
        return grad_x, None, None


class _UnshiftUnpartitionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, windows, B, D, H, W, C, window_size, shift_size):
        windows = windows.contiguous()
        out = torch.empty((B, D, H, W, C), dtype=windows.dtype, device=windows.device)
        _ops.fused_unshift_unpartition_3d(
            out,
            windows,
            B,
            D,
            H,
            W,
            C,
            *window_size,
            *shift_size,
        )
        ctx.window_size = window_size
        ctx.shift_size = shift_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        B, D, H, W, C = (int(v) for v in grad_output.shape)
        ws_d, ws_h, ws_w = ctx.window_size
        volume = ws_d * ws_h * ws_w
        num_windows = (D // ws_d) * (H // ws_h) * (W // ws_w)
        grad_windows = torch.empty(
            (B * num_windows, volume, C),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        _ops.fused_unshift_unpartition_3d_backward(
            grad_windows,
            grad_output.contiguous(),
            *ctx.window_size,
            *ctx.shift_size,
        )
        return grad_windows, None, None, None, None, None, None, None


def fused_shift_partition_3d(
    x: torch.Tensor,
    window_size: Sequence[int],
    shift_size: Sequence[int],
) -> torch.Tensor:
    """Fuse cyclic shift and 3D window partition for `[B, D, H, W, C]`."""

    if x.ndim != 5:
        raise ValueError("x must have shape [B, D, H, W, C]")
    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_dtype(x.dtype)
    B, D, H, W, C = (int(v) for v in x.shape)
    _validate_shape(D, H, W, window_size, shift_size)
    volume = window_size[0] * window_size[1] * window_size[2]
    num_windows = (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])

    if x.device.type == "meta":
        return torch.empty((B * num_windows, volume, C), dtype=x.dtype, device=x.device)
    if not any(shift_size):
        return window_partition_reference(x, window_size)
    if _can_use_cuda_tensor(x):
        with _nvtx_range("crumpet.fused_shift_partition_3d"):
            if torch.is_grad_enabled() and x.requires_grad:
                return _ShiftPartitionFn.apply(x, window_size, shift_size)
            out = torch.empty((B * num_windows, volume, C), dtype=x.dtype, device=x.device)
            _ops.fused_shift_partition_3d(
                out,
                x.contiguous(),
                *window_size,
                *shift_size,
            )
            return out
    return unfused_shift_partition_3d_reference(x, window_size, shift_size)


def fused_unshift_unpartition_3d(
    windows: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
    C: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
) -> torch.Tensor:
    """Fuse 3D window reverse and reverse cyclic shift."""

    if windows.ndim != 3:
        raise ValueError("windows must have shape [B * num_windows, window_volume, C]")
    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_dtype(windows.dtype)
    B, D, H, W, C = (int(B), int(D), int(H), int(W), int(C))
    _validate_shape(D, H, W, window_size, shift_size)
    ws_d, ws_h, ws_w = window_size
    volume = ws_d * ws_h * ws_w
    num_windows = (D // ws_d) * (H // ws_h) * (W // ws_w)
    expected = (B * num_windows, volume, C)
    if tuple(int(v) for v in windows.shape) != expected:
        raise ValueError(f"windows must have shape {expected}")

    if windows.device.type == "meta":
        return torch.empty((B, D, H, W, C), dtype=windows.dtype, device=windows.device)
    if not any(shift_size):
        return window_reverse_reference(windows, window_size, (B, D, H, W))
    if _can_use_cuda_tensor(windows):
        with _nvtx_range("crumpet.fused_unshift_unpartition_3d"):
            if torch.is_grad_enabled() and windows.requires_grad:
                return _UnshiftUnpartitionFn.apply(
                    windows,
                    B,
                    D,
                    H,
                    W,
                    C,
                    window_size,
                    shift_size,
                )
            out = torch.empty((B, D, H, W, C), dtype=windows.dtype, device=windows.device)
            _ops.fused_unshift_unpartition_3d(
                out,
                windows.contiguous(),
                B,
                D,
                H,
                W,
                C,
                *window_size,
                *shift_size,
            )
            return out
    return unfused_unshift_unpartition_3d_reference(
        windows, B, D, H, W, C, window_size, shift_size
    )


def patch_monai_swin_unetr() -> bool:
    from .monai_patch import patch_monai_swin_unetr as patch

    return patch()


def unpatch_monai_swin_unetr() -> bool:
    from .monai_patch import unpatch_monai_swin_unetr as unpatch

    return unpatch()


__all__ = [
    "compute_attn_mask_3d",
    "fused_shift_partition_3d",
    "fused_unshift_unpartition_3d",
    "patch_monai_swin_unetr",
    "unpatch_monai_swin_unetr",
    "clear_mask_cache",
    "mask_cache_info",
    "window_partition_reference",
    "window_reverse_reference",
    "unfused_shift_partition_3d_reference",
    "unfused_unshift_unpartition_3d_reference",
]

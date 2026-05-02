"""Pure PyTorch reference functions for CRUMPET.

These functions follow the MONAI Swin UNETR window code and the Microsoft Swin
Transformer window partition convention. They are used for correctness tests,
CPU fallback paths and baseline benchmarks.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _triple(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    out = tuple(int(v) for v in value)
    if any(v < 0 for v in out):
        raise ValueError(f"{name} values must be non-negative")
    return out


def _check_divisible(
    D: int,
    H: int,
    W: int,
    window_size: tuple[int, int, int],
) -> None:
    ws_d, ws_h, ws_w = window_size
    if ws_d <= 0 or ws_h <= 0 or ws_w <= 0:
        raise ValueError("window_size values must be positive")
    if D % ws_d or H % ws_h or W % ws_w:
        raise ValueError(
            "standalone CRUMPET kernels require D, H and W to be divisible by window_size"
        )


def _check_shift(
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> None:
    for axis, (shift, window) in enumerate(zip(shift_size, window_size)):
        if shift < 0 or shift >= window:
            raise ValueError(
                f"shift_size[{axis}] must satisfy 0 <= shift < window_size[{axis}]"
            )


def compute_attn_mask_3d_reference(
    D: int,
    H: int,
    W: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return MONAI-compatible shifted-window attention masks.

    MONAI builds integer region labels with Python slices, partitions those
    labels into windows and compares every token pair within a window. This is
    the same integer procedure expressed with vectorised PyTorch operations.
    """

    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_divisible(D, H, W, window_size)
    _check_shift(window_size, shift_size)
    ws_d, ws_h, ws_w = window_size
    ss_d, ss_h, ss_w = shift_size
    volume = ws_d * ws_h * ws_w
    num_windows = (D // ws_d) * (H // ws_h) * (W // ws_w)

    if ss_d == 0 and ss_h == 0 and ss_w == 0:
        return torch.zeros((num_windows, volume, volume), dtype=dtype, device=device)

    d = torch.arange(D, device=device)
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)

    region_d = torch.where(d < D - ws_d, 0, torch.where(d < D - ss_d, 1, 2))
    region_h = torch.where(h < H - ws_h, 0, torch.where(h < H - ss_h, 1, 2))
    region_w = torch.where(w < W - ws_w, 0, torch.where(w < W - ss_w, 1, 2))

    labels = (
        9 * region_d[:, None, None]
        + 3 * region_h[None, :, None]
        + region_w[None, None, :]
    ).to(torch.int16)
    labels = labels.reshape(1, D, H, W, 1)
    windows = window_partition_reference(labels, window_size).squeeze(-1)
    mask = windows.unsqueeze(1) != windows.unsqueeze(2)
    return torch.where(
        mask,
        torch.tensor(-100.0, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    )


def window_partition_reference(
    x: torch.Tensor,
    window_size: Sequence[int],
) -> torch.Tensor:
    """Partition `[B, D, H, W, C]` into MONAI/Swin 3D windows."""

    if x.ndim != 5:
        raise ValueError("window_partition_reference expects [B, D, H, W, C]")
    window_size = _triple("window_size", window_size)
    B, D, H, W, C = x.shape
    ws_d, ws_h, ws_w = window_size
    _check_divisible(D, H, W, window_size)
    x = x.view(B, D // ws_d, ws_d, H // ws_h, ws_h, W // ws_w, ws_w, C)
    return (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, ws_d * ws_h * ws_w, C)
    )


def window_reverse_reference(
    windows: torch.Tensor,
    window_size: Sequence[int],
    dims: Sequence[int],
) -> torch.Tensor:
    """Reverse MONAI/Swin 3D window partitioning."""

    window_size = _triple("window_size", window_size)
    if len(dims) == 4:
        B, D, H, W = (int(v) for v in dims)
    elif len(dims) == 5:
        B, D, H, W, _ = (int(v) for v in dims)
    else:
        raise ValueError("dims must be [B, D, H, W] or [B, D, H, W, C]")
    ws_d, ws_h, ws_w = window_size
    _check_divisible(D, H, W, window_size)
    x = windows.view(B, D // ws_d, H // ws_h, W // ws_w, ws_d, ws_h, ws_w, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


def unfused_shift_partition_3d_reference(
    x: torch.Tensor,
    window_size: Sequence[int],
    shift_size: Sequence[int],
) -> torch.Tensor:
    """Reference `torch.roll(..., shifts=-shift_size)` plus window partition."""

    if x.ndim != 5:
        raise ValueError("x must have shape [B, D, H, W, C]")
    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_divisible(int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), window_size)
    _check_shift(window_size, shift_size)
    if any(shift_size):
        x = torch.roll(
            x,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(1, 2, 3),
        )
    return window_partition_reference(x, window_size)


def unfused_unshift_unpartition_3d_reference(
    windows: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
    C: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
) -> torch.Tensor:
    """Reference window reverse plus `torch.roll(..., shifts=shift_size)`."""

    window_size = _triple("window_size", window_size)
    shift_size = _triple("shift_size", shift_size)
    _check_divisible(D, H, W, window_size)
    _check_shift(window_size, shift_size)
    x = window_reverse_reference(windows, window_size, (B, D, H, W))
    if int(x.shape[-1]) != int(C):
        raise ValueError("C does not match the windows tensor")
    if any(shift_size):
        x = torch.roll(x, shifts=shift_size, dims=(1, 2, 3))
    return x


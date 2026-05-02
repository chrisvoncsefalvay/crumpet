"""Reversible MONAI Swin UNETR patching.

Replaces three internal forward paths inside MONAI's `SwinUNETR`:

  * `SwinTransformerBlock.forward_part1` — the LayerNorm + cyclic-shift +
    window-partition + attention + window-reverse + reverse-roll wrapper
    routes through `fused_shift_partition_3d` and
    `fused_unshift_unpartition_3d`.
  * `BasicLayer.forward` — wires the channel-last permutation and uses
    the cached `compute_attn_mask_3d` for the shifted attention mask.
  * `WindowAttention.forward` — replaces the eager `q @ k.T + bias + mask
    -> softmax -> attn @ v` chain with the Triton fused FlashAttention-2
    kernel from `crumpet.fused_attention`. Falls back to the eager path
    during training, with non-zero attention dropout, or when the head
    dimension is below the Triton `tl.dot` minimum (16).

`patch_monai_swin_unetr` is idempotent and `unpatch_monai_swin_unetr`
restores the originals so the same model can be benchmarked under both
paths.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch
import torch.nn.functional as F

from . import (
    compute_attn_mask_3d,
    fused_shift_partition_3d,
    fused_unshift_unpartition_3d,
)

try:
    from .fused_attention import fused_swin_attention as _fused_swin_attention
    _HAS_FUSED_ATTN = True
except Exception:
    _fused_swin_attention = None
    _HAS_FUSED_ATTN = False


_STATE: dict[str, Any] = {}


def _fused_attn_enabled() -> bool:
    """Allow opt-out via `CRUMPET_DISABLE_FUSED_ATTN=1`."""
    if not _HAS_FUSED_ATTN:
        return False
    return os.environ.get("CRUMPET_DISABLE_FUSED_ATTN", "0") != "1"


def _load_monai_swin():
    try:
        import monai.networks.nets.swin_unetr as swin
    except Exception as exc:
        raise RuntimeError("MONAI Swin UNETR is not importable") from exc
    for name in ("SwinTransformerBlock", "BasicLayer", "WindowAttention", "get_window_size"):
        if not hasattr(swin, name):
            raise RuntimeError(f"installed MONAI is missing {name}")
    return swin


def _as_triple(value) -> tuple[int, int, int]:
    if len(value) != 3:
        raise RuntimeError("CRUMPET MONAI patch only supports 3D Swin blocks")
    return tuple(int(v) for v in value)


def _patched_forward_part1(original, swin):
    @functools.wraps(original)
    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        if len(x_shape) != 5:
            return original(self, x, mask_matrix)

        x = self.norm1(x)
        b, d, h, w, c = x.shape
        window_size, shift_size = swin.get_window_size((d, h, w), self.window_size, self.shift_size)
        window_size = _as_triple(window_size)
        shift_size = _as_triple(shift_size)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            attn_mask = mask_matrix
        else:
            attn_mask = None

        x_windows = fused_shift_partition_3d(x, window_size, shift_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        x = fused_unshift_unpartition_3d(
            attn_windows,
            B=b,
            D=dp,
            H=hp,
            W=wp,
            C=c,
            window_size=window_size,
            shift_size=shift_size,
        )

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    return forward_part1


def _patched_window_attention_forward(original):
    """Replace the MONAI ``WindowAttention.forward`` with a fused path.

    The eager chain materialises the `[B * num_windows, num_heads, N, N]`
    attention-scores tensor four times — as the BMM output, after the
    relative-position-bias add, after the shifted-window mask add, and
    after softmax. At BTCV stage 0 (B=2744, H=3, N=343) that intermediate
    is 1.94 GB; reading and re-writing it dominated the call. The fused
    Triton kernel keeps the scores in registers, applies bias and mask
    inside the inner loop, and only writes the `[B, H, N, D]` output.

    This path keeps the original qkv and proj linears (cuBLAS GEMMs that
    are already well tuned) and replaces only the attention compute
    between them.
    """

    @functools.wraps(original)
    def forward(self, x, mask):
        # Fall back to eager during training so gradients flow through the
        # learnable relative-position bias and any non-zero attn dropout.
        if self.training or self.attn_drop.p != 0.0:
            return original(self, x, mask)
        if not _fused_attn_enabled():
            return original(self, x, mask)

        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        # Triton's `tl.dot` requires the contraction dimension >= 16. The
        # production MONAI Swin UNETR uses head_dim = 16 (feature_size 48
        # / num_heads (3, 6, 12, 24)); the synthetic smoke model uses
        # head_dim = 4 and goes through the eager fallback.
        if head_dim < 16:
            return original(self, x, mask)

        qkv = self.qkv(x).reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, H, n, D]

        # Recompute the per-head relative-position bias every call rather
        # than caching on the module. The gather + permute is sub-
        # millisecond on a tensor of at most ~ 1 MB and is fully
        # CUDA-graph capturable, which a cached attribute would not be.
        rel_pos_idx = self.relative_position_index[:n, :n].reshape(-1)
        rel_bias = self.relative_position_bias_table[rel_pos_idx].reshape(n, n, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous().to(dtype=x.dtype)

        # MONAI passes mask as `[num_windows_per_batch, n, n]` with
        # `b = outer_batch * num_windows_per_batch`. The fused kernel
        # consumes a per-batch mask, so for outer_batch > 1 we tile.
        if mask is not None:
            nw = mask.shape[0]
            outer = b // nw
            if outer == 1:
                mask_for_kernel = mask
            else:
                mask_for_kernel = mask.repeat(outer, 1, 1)

            # Per-window boundary flag — interior windows have an all-
            # zero mask and the kernel skips the mask read for them.
            flat = mask_for_kernel.view(mask_for_kernel.shape[0], -1)
            boundary = (flat.abs().amax(dim=1) > 0).to(torch.uint8).contiguous()
        else:
            mask_for_kernel = None
            boundary = None

        out = _fused_swin_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            rel_bias,
            mask_for_kernel,
            boundary,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return forward


def _patched_basic_layer_forward(original, swin):
    @functools.wraps(original)
    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) != 5:
            return original(self, x)

        b, c, d, h, w = x_shape
        window_size, shift_size = swin.get_window_size((d, h, w), self.window_size, self.shift_size)
        window_size = _as_triple(window_size)
        shift_size = _as_triple(shift_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        dp = ((d + window_size[0] - 1) // window_size[0]) * window_size[0]
        hp = ((h + window_size[1] - 1) // window_size[1]) * window_size[1]
        wp = ((w + window_size[2] - 1) // window_size[2]) * window_size[2]
        attn_mask = compute_attn_mask_3d(
            D=dp,
            H=hp,
            W=wp,
            window_size=window_size,
            shift_size=shift_size,
            dtype=x.dtype,
            device=x.device,
        )
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()

    return forward


def patch_monai_swin_unetr() -> bool:
    """Patch MONAI 3D Swin block internals.

    Returns True on the call that installs the patch, False if the patch
    is already active. Idempotent.
    """

    if _STATE.get("patched"):
        return False
    swin = _load_monai_swin()
    original_part1 = swin.SwinTransformerBlock.forward_part1
    original_layer_forward = swin.BasicLayer.forward
    original_window_attn = swin.WindowAttention.forward
    _STATE.update(
        {
            "patched": True,
            "swin": swin,
            "forward_part1": original_part1,
            "basic_layer_forward": original_layer_forward,
            "window_attention_forward": original_window_attn,
        }
    )
    swin.SwinTransformerBlock.forward_part1 = _patched_forward_part1(original_part1, swin)
    swin.BasicLayer.forward = _patched_basic_layer_forward(original_layer_forward, swin)
    if _fused_attn_enabled():
        swin.WindowAttention.forward = _patched_window_attention_forward(original_window_attn)
    return True


def unpatch_monai_swin_unetr() -> bool:
    """Undo :func:`patch_monai_swin_unetr`."""

    if not _STATE.get("patched"):
        return False
    swin = _STATE["swin"]
    swin.SwinTransformerBlock.forward_part1 = _STATE["forward_part1"]
    swin.BasicLayer.forward = _STATE["basic_layer_forward"]
    if "window_attention_forward" in _STATE:
        swin.WindowAttention.forward = _STATE["window_attention_forward"]
    _STATE.clear()
    return True

"""Triton fused 3D Swin window attention.

Fuses scaled-dot-product attention with the per-head relative-position
bias and the per-window shifted attention mask into a single Triton
kernel built around the FlashAttention-2 online-softmax recurrence.

Compared to the eager implementation in MONAI's ``WindowAttention``
the fused kernel:

- Never materialises the ``[B, H, N, N]`` attention scores tensor.
  At the BTCV stage-0 shape (B=2744 windows, H=3, N=343) that
  intermediate is 1.94 GB; the eager path reads and re-writes it
  three times (BMM output, +bias, +mask, softmax).
- Never materialises a combined ``bias + mask`` tensor. The relative
  position bias is supplied as ``[H, N, N]`` and broadcast across
  batch; the per-window shifted mask is supplied as
  ``[num_windows, N, N]`` and applied in the inner loop on
  boundary windows only — interior windows skip the mask read
  entirely via the ``boundary`` flag.
- Writes a single ``[B, H, N, D]`` output (~ 90 MB at BTCV stage 0).

Per-call wall clock on Spark GB10 (fp16, BTCV shapes):

    stage     B    H    N      eager     fused    speedup
    0       2744   3   343   91.3 ms   12.4 ms     7.4x
    1        343   6   343   22.8 ms    3.1 ms     7.4x
    2         64  12   343    8.7 ms    1.2 ms     7.3x
    3          8  24   343    2.3 ms    0.27 ms    8.5x

NCU reports the kernel at 78 % of the GB10 compute peak and 78 % of
the memory peak with the activity classification "Compute and Memory
are well-balanced".

The launcher dispatches a 3-D grid ``(ceil(N / BLOCK_M), H, B)`` so
each CTA owns one ``(batch, head, query-row-tile)`` triple. Tile sizes
``BLOCK_M = 64`` and ``BLOCK_N = 32`` were chosen by sweep on Spark;
larger ``BLOCK_N`` exceeded the 100 KB shared-memory cap on this
architecture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_swin_attention_kernel(
    Q, K, V,
    BIAS, MASK, BOUNDARY,
    OUT,
    SCALE,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_bh, stride_bm, stride_bn,
    stride_mb, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One CTA owns one ``(batch, head, query-row-tile)`` triple.

    The CTA streams over the K / V dimension in ``BLOCK_N`` chunks, applies
    the FlashAttention-2 online-softmax recurrence and writes the final
    normalised output. The relative-position bias broadcasts across the
    batch dimension; the shifted mask is read only for boundary windows.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    is_boundary = tl.load(BOUNDARY + pid_b)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N
    offs_d = tl.arange(0, D)

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh \
            + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptr, mask=m_mask[:, None], other=0.0)
    # Pre-multiply by the softmax scale in fp32 to avoid range issues, then
    # cast back to the input dtype so the operands of `tl.dot` agree. The
    # dot accumulator is fp32 regardless of operand precision.
    q = (q.to(tl.float32) * SCALE).to(q.dtype)

    # FA2 online-softmax accumulators (running max, running denominator,
    # unnormalised output).
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    bias_base = BIAS + pid_h * stride_bh + offs_m[:, None] * stride_bm
    mask_base = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mm

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k_ptr = K + pid_b * stride_kb + pid_h * stride_kh \
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptr = V + pid_b * stride_vb + pid_h * stride_vh \
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptr, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr, mask=n_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        # Per-head relative-position bias (broadcast across batch).
        bias_blk = tl.load(
            bias_base + offs_n[None, :] * stride_bn,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        scores = scores + bias_blk.to(tl.float32)

        # Per-window shifted mask. Interior windows have an all-zero mask
        # so we skip the load to avoid 645 MB of pointless DRAM traffic
        # at BTCV stage 0.
        if is_boundary != 0:
            mask_blk = tl.load(
                mask_base + offs_n[None, :] * stride_mn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            scores = scores + mask_blk.to(tl.float32)

        # Mask out padded N positions so they don't influence softmax.
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # Online softmax recurrence (FlashAttention-2).
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptr = OUT + pid_b * stride_ob + pid_h * stride_oh \
              + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptr, acc.to(OUT.dtype.element_ty), mask=m_mask[:, None])


def fused_swin_attention(
    q: torch.Tensor,                            # [B, H, N, D]
    k: torch.Tensor,                            # [B, H, N, D]
    v: torch.Tensor,                            # [B, H, N, D]
    relative_position_bias: torch.Tensor,       # [H, N, N]
    mask: torch.Tensor | None,                  # [num_windows, N, N] or None
    boundary: torch.Tensor | None = None,       # [B] uint8 / bool
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 32,
) -> torch.Tensor:
    """Compute fused 3D Swin window attention.

    `q`, `k`, `v` have shape ``[B * num_windows, num_heads, N, head_dim]``.
    `relative_position_bias` is ``[num_heads, N, N]`` and broadcasts across
    the leading batch dimension. `mask` is the per-window shifted mask
    ``[num_windows, N, N]``; it may be ``None`` for the unshifted layer.

    `boundary` is a per-batch ``uint8`` flag (0 = interior, 1 = boundary).
    When supplied, the kernel skips the per-window mask read for interior
    windows. When omitted, every window is treated as boundary; the
    integration layer in ``crumpet.monai_patch`` constructs and passes
    the flag from the cached attention mask.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape, "Q/K/V shape mismatch"
    B, H, N, D = q.shape
    assert relative_position_bias.shape == (H, N, N)

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # When no mask is supplied, fabricate an unused single-row tensor and
    # mark every window as interior so the kernel skips the mask load.
    if mask is None:
        mask_t = torch.zeros((1, N, N), device=q.device, dtype=q.dtype)
        bnd = torch.zeros((B,), device=q.device, dtype=torch.uint8)
        mask_strides = (0, mask_t.stride(1), mask_t.stride(2))
    else:
        mask_t = mask
        # MONAI passes one mask per window in the leading batch dim.
        # When B is exactly num_windows we reuse the strides as-is;
        # the broadcast-by-1 case sets the leading stride to 0.
        if mask_t.shape[0] == 1:
            mask_strides = (0, mask_t.stride(1), mask_t.stride(2))
        else:
            assert mask_t.shape[0] == B, \
                f"mask shape {mask_t.shape[0]} incompatible with B={B}"
            mask_strides = (mask_t.stride(0), mask_t.stride(1), mask_t.stride(2))

        if boundary is None:
            bnd = torch.ones((B,), device=q.device, dtype=torch.uint8)
        else:
            bnd = boundary if boundary.dtype == torch.uint8 else boundary.to(torch.uint8)

    out = torch.empty_like(q)

    block_n = min(block_n, triton.next_power_of_2(N))
    block_m = min(block_m, triton.next_power_of_2(N))

    grid = (triton.cdiv(N, block_m), H, B)
    _fused_swin_attention_kernel[grid](
        q, k, v,
        relative_position_bias, mask_t, bnd,
        out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        relative_position_bias.stride(0), relative_position_bias.stride(1), relative_position_bias.stride(2),
        mask_strides[0], mask_strides[1], mask_strides[2],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N=N,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


__all__ = ["fused_swin_attention"]

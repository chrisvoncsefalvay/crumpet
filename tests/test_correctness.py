from __future__ import annotations

import itertools

import pytest
import torch

import crumpet
from crumpet.reference import (
    compute_attn_mask_3d_reference,
    unfused_shift_partition_3d_reference,
    unfused_unshift_unpartition_3d_reference,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    ("shape", "window_size", "shift_size"),
    [
        ((14, 14, 14), (7, 7, 7), (3, 3, 3)),
        ((14, 14, 14), (7, 7, 7), (0, 0, 0)),
        ((8, 8, 8), (4, 4, 4), (2, 2, 2)),
        ((6, 4, 2), (2, 2, 2), (1, 1, 1)),
    ],
)
def test_mask_matches_reference(shape, window_size, shift_size):
    device = _device()
    got = crumpet.compute_attn_mask_3d(
        *shape,
        window_size=window_size,
        shift_size=shift_size,
        dtype=torch.float32,
        device=device,
    )
    ref = compute_attn_mask_3d_reference(
        *shape,
        window_size=window_size,
        shift_size=shift_size,
        dtype=torch.float32,
        device=device,
    )
    assert torch.equal(got, ref)


def test_shift_size_zero_mask_is_zero():
    mask = crumpet.compute_attn_mask_3d(
        14,
        14,
        14,
        window_size=(7, 7, 7),
        shift_size=(0, 0, 0),
        dtype=torch.float16,
        device=_device(),
    )
    assert torch.count_nonzero(mask).item() == 0


@pytest.mark.parametrize(
    ("shape", "window_size", "shift_size", "C"),
    [
        ((14, 14, 14), (7, 7, 7), (3, 3, 3), 13),
        ((14, 14, 14), (7, 7, 7), (0, 0, 0), 48),
        ((8, 8, 8), (4, 4, 4), (2, 2, 2), 7),
    ],
)
def test_partition_unpartition_match_reference(shape, window_size, shift_size, C):
    device = _device()
    x = torch.randn((2, *shape, C), device=device, dtype=torch.float32)
    got = crumpet.fused_shift_partition_3d(x, window_size, shift_size)
    ref = unfused_shift_partition_3d_reference(x, window_size, shift_size)
    assert torch.equal(got, ref)

    restored = crumpet.fused_unshift_unpartition_3d(
        got,
        B=2,
        D=shape[0],
        H=shape[1],
        W=shape[2],
        C=C,
        window_size=window_size,
        shift_size=shift_size,
    )
    ref_restored = unfused_unshift_unpartition_3d_reference(
        ref,
        B=2,
        D=shape[0],
        H=shape[1],
        W=shape[2],
        C=C,
        window_size=window_size,
        shift_size=shift_size,
    )
    assert torch.equal(restored, ref_restored)
    assert torch.equal(restored, x)


def test_partition_mapping_is_bijective():
    D, H, W = 14, 14, 14
    ws = (7, 7, 7)
    ss = (3, 3, 3)
    seen = set()
    for win_d, win_h, win_w in itertools.product(range(2), repeat=3):
        for loc_d, loc_h, loc_w in itertools.product(range(7), repeat=3):
            src = (
                (win_d * ws[0] + loc_d + ss[0]) % D,
                (win_h * ws[1] + loc_h + ss[1]) % H,
                (win_w * ws[2] + loc_w + ss[2]) % W,
            )
            assert src not in seen
            seen.add(src)
    assert len(seen) == D * H * W


from __future__ import annotations

import torch

import crumpet


def test_meta_shape_propagation():
    x = torch.empty((2, 14, 14, 14, 48), device="meta", dtype=torch.float16)
    windows = crumpet.fused_shift_partition_3d(x, (7, 7, 7), (3, 3, 3))
    assert windows.shape == (16, 343, 48)
    restored = crumpet.fused_unshift_unpartition_3d(
        windows,
        2,
        14,
        14,
        14,
        48,
        (7, 7, 7),
        (3, 3, 3),
    )
    assert restored.shape == x.shape
    mask = crumpet.compute_attn_mask_3d(
        14,
        14,
        14,
        (7, 7, 7),
        (3, 3, 3),
        torch.float16,
        "meta",
    )
    assert mask.shape == (8, 343, 343)


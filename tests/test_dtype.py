from __future__ import annotations

import pytest
import torch

import crumpet


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_supported_dtypes(dtype):
    device = _device()
    if device.type == "cpu" and dtype == torch.float16:
        pytest.skip("CPU fp16 arithmetic is not useful for this fallback test")
    x = torch.randn((1, 8, 8, 8, 4), device=device, dtype=dtype)
    y = crumpet.fused_shift_partition_3d(x, (4, 4, 4), (2, 2, 2))
    z = crumpet.fused_unshift_unpartition_3d(y, 1, 8, 8, 8, 4, (4, 4, 4), (2, 2, 2))
    assert y.dtype == dtype
    assert z.dtype == dtype
    assert torch.equal(z, x)
    mask = crumpet.compute_attn_mask_3d(8, 8, 8, (4, 4, 4), (2, 2, 2), dtype, device)
    assert mask.dtype == dtype


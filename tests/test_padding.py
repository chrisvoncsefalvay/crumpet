from __future__ import annotations

import pytest
import torch

import crumpet


def test_invalid_divisibility_fails_clearly():
    x = torch.randn((1, 15, 14, 14, 4))
    with pytest.raises(ValueError, match="divisible"):
        crumpet.fused_shift_partition_3d(x, (7, 7, 7), (3, 3, 3))


def test_invalid_shift_fails_clearly():
    x = torch.randn((1, 14, 14, 14, 4))
    with pytest.raises(ValueError, match="shift_size"):
        crumpet.fused_shift_partition_3d(x, (7, 7, 7), (7, 0, 0))


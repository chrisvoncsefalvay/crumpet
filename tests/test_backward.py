from __future__ import annotations

import pytest
import torch

import crumpet
from crumpet.reference import unfused_shift_partition_3d_reference


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("shift_size", [(3, 3, 3), (0, 0, 0)])
def test_shift_partition_backward_matches_reference(shift_size):
    device = _device()
    x = torch.randn((1, 14, 14, 14, 5), device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    y = crumpet.fused_shift_partition_3d(x, (7, 7, 7), shift_size)
    y_ref = unfused_shift_partition_3d_reference(x_ref, (7, 7, 7), shift_size)
    grad = torch.randn_like(y)
    y.backward(grad)
    y_ref.backward(grad)
    assert torch.equal(x.grad, x_ref.grad)


def test_reference_gradcheck_tiny():
    x = torch.randn((1, 4, 4, 4, 2), dtype=torch.double, requires_grad=True)

    def fn(inp):
        return unfused_shift_partition_3d_reference(inp, (2, 2, 2), (1, 1, 1))

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)


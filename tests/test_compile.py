from __future__ import annotations

import pytest
import torch

import crumpet


def test_torch_compile_fullgraph_smoke():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    class Module(torch.nn.Module):
        def forward(self, x):
            w = crumpet.fused_shift_partition_3d(x, (2, 2, 2), (1, 1, 1))
            return crumpet.fused_unshift_unpartition_3d(w, x.shape[0], 4, 4, 4, x.shape[-1], (2, 2, 2), (1, 1, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 4, 4, 4, 3), device=device)
    compiled = torch.compile(Module(), fullgraph=True)
    assert torch.equal(compiled(x), x)


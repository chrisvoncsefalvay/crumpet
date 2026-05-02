from __future__ import annotations

import json

import pytest
import torch


def test_demo_btcv_synthetic_smoke(tmp_path):
    pytest.importorskip("monai")
    if not torch.cuda.is_available():
        pytest.skip("synthetic BTCV smoke uses CUDA timing")

    from crumpet.demo_btcv import main

    output = tmp_path / "btcv_demo.json"
    main(
        [
            "--device",
            "cuda:0",
            "--dtype",
            "fp16",
            "--roi-size",
            "64",
            "64",
            "64",
            "--sw-batch-size",
            "1",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--output-json",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    assert payload["synthetic"] is True
    assert payload["max_abs_diff"] == 0.0


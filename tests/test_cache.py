from __future__ import annotations

import torch

from crumpet.cache import (
    clear_mask_cache,
    get_or_compute_mask,
    make_mask_key,
    mask_cache_info,
)


def test_cache_hit_and_miss(monkeypatch):
    clear_mask_cache()
    calls = {"n": 0}
    key = make_mask_key(14, 14, 14, (7, 7, 7), (3, 3, 3), torch.float32, "cpu")

    def compute():
        calls["n"] += 1
        return torch.zeros((1,), dtype=torch.float32)

    a = get_or_compute_mask(key, compute)
    b = get_or_compute_mask(key, compute)
    assert a is b
    assert calls["n"] == 1
    assert mask_cache_info()["entries"] == 1


def test_cache_disabled(monkeypatch):
    clear_mask_cache()
    monkeypatch.setenv("CRUMPET_DISABLE_MASK_CACHE", "1")
    key = make_mask_key(14, 14, 14, (7, 7, 7), (3, 3, 3), torch.float32, "cpu")
    a = get_or_compute_mask(key, lambda: torch.zeros((1,)))
    b = get_or_compute_mask(key, lambda: torch.ones((1,)))
    assert not torch.equal(a, b)


def test_cache_byte_cap(monkeypatch):
    clear_mask_cache()
    monkeypatch.setenv("CRUMPET_MASK_CACHE_MAX_BYTES", "1")
    key = make_mask_key(14, 14, 14, (7, 7, 7), (3, 3, 3), torch.float32, "cpu")
    get_or_compute_mask(key, lambda: torch.zeros((2,), dtype=torch.float32))
    assert mask_cache_info()["entries"] == 0


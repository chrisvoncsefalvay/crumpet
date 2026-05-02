"""Device-aware shifted attention mask cache."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable

import torch

MaskKey = tuple[int, int, int, tuple[int, int, int], tuple[int, int, int], torch.dtype, str]

_CACHE: "OrderedDict[MaskKey, torch.Tensor]" = OrderedDict()
_CACHE_BYTES = 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _disabled() -> bool:
    return os.environ.get("CRUMPET_DISABLE_MASK_CACHE", "0") == "1"


def _device_key(device: torch.device | str) -> str:
    dev = torch.device(device)
    if dev.type == "cuda":
        index = dev.index
        if index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return f"cuda:{index if index is not None else 0}"
    return str(dev)


def make_mask_key(
    D: int,
    H: int,
    W: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> MaskKey:
    return (D, H, W, window_size, shift_size, dtype, _device_key(device))


def clear_mask_cache() -> None:
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0


def mask_cache_info() -> dict[str, int]:
    return {
        "entries": len(_CACHE),
        "bytes": _CACHE_BYTES,
        "max_entries": _env_int("CRUMPET_MASK_CACHE_SIZE", 64),
        "max_bytes": _env_int("CRUMPET_MASK_CACHE_MAX_BYTES", 1073741824),
    }


def get_or_compute_mask(
    key: MaskKey,
    compute: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Return a cached mask or compute it.

    Callers in hot loops should keep their own reference to the returned tensor.
    The cache is a convenience LRU, not a lifetime guarantee.
    """

    global _CACHE_BYTES
    if _disabled():
        return compute()

    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    value = compute()
    nbytes = value.numel() * value.element_size()
    max_bytes = _env_int("CRUMPET_MASK_CACHE_MAX_BYTES", 1073741824)
    max_entries = _env_int("CRUMPET_MASK_CACHE_SIZE", 64)
    if nbytes > max_bytes or max_entries <= 0:
        return value

    _CACHE[key] = value
    _CACHE_BYTES += nbytes
    while len(_CACHE) > max_entries or _CACHE_BYTES > max_bytes:
        _, evicted = _CACHE.popitem(last=False)
        _CACHE_BYTES -= evicted.numel() * evicted.element_size()
    return value


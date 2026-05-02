#!/usr/bin/env python3
"""Summarise CRUMPET benchmark JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    for path in sorted(Path("benchmarks/results").glob("*.json")):
        data = json.loads(path.read_text())
        print(path)
        for key in (
            "go_no_go_decision",
            "large_case_speedup",
            "large_shift_partition_speedup",
            "speedup",
            "max_abs_diff",
            "halt_rule_triggered",
        ):
            if key in data:
                print(f"  {key}: {data[key]}")
        if "roofline" in data:
            print(f"  roofline_status: {data['roofline']['status']}")
            print(f"  roofline_reason: {data['roofline']['reason']}")
        if "nsight_systems" in data:
            nsys = data["nsight_systems"]
            shift = nsys["shift_partition_kernel_half_shifted"]["avg_ns"] / 1_000_000
            unshift = nsys["unshift_unpartition_kernel_half_shifted"]["avg_ns"] / 1_000_000
            mask = nsys["attn_mask_kernel_half_no_cache"]["avg_ns"] / 1_000_000
            memset = nsys["attn_mask_memset_half_no_cache"]["avg_ns"] / 1_000_000
            print(f"  nsys_shift_partition_avg_ms: {shift:.6f}")
            print(f"  nsys_unshift_unpartition_avg_ms: {unshift:.6f}")
            print(f"  nsys_mask_kernel_no_cache_avg_ms: {mask:.6f}")
            print(f"  nsys_mask_memset_no_cache_avg_ms: {memset:.6f}")


if __name__ == "__main__":
    main()

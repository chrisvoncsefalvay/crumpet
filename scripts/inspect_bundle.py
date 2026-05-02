#!/usr/bin/env python3
"""Inspect a MONAI Swin UNETR BTCV bundle directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir")
    args = parser.parse_args()
    root = Path(args.bundle_dir)
    files = {
        "root": root.exists(),
        "models/model.pt": (root / "models" / "model.pt").exists(),
        "configs/inference.json": (root / "configs" / "inference.json").exists(),
        "configs/metadata.json": (root / "configs" / "metadata.json").exists(),
    }
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()


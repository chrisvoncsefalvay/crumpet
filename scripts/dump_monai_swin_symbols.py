#!/usr/bin/env python3
"""Print MONAI Swin UNETR symbols relevant to CRUMPET patching."""

from __future__ import annotations

import inspect
import json


def main() -> None:
    try:
        import monai
        import monai.networks.nets.swin_unetr as swin
    except Exception as exc:
        print(json.dumps({"available": False, "error": repr(exc)}, indent=2))
        return

    names = [
        "window_partition",
        "window_reverse",
        "compute_mask",
        "get_window_size",
        "SwinTransformerBlock",
        "BasicLayer",
    ]
    symbols = {}
    for name in names:
        obj = getattr(swin, name, None)
        if obj is None:
            symbols[name] = {"available": False}
            continue
        target = obj.forward_part1 if name == "SwinTransformerBlock" else obj
        target = obj.forward if name == "BasicLayer" else target
        symbols[name] = {
            "available": True,
            "module": getattr(obj, "__module__", None),
            "file": inspect.getsourcefile(obj),
            "line": inspect.getsourcelines(target)[1],
        }
    print(
        json.dumps(
            {
                "available": True,
                "monai_version": monai.__version__,
                "symbols": symbols,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


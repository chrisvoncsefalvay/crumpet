from __future__ import annotations

import pytest

import crumpet


def test_patch_unpatch_repeated_safe():
    pytest.importorskip("monai")
    assert crumpet.patch_monai_swin_unetr() is True
    assert crumpet.patch_monai_swin_unetr() is False
    assert crumpet.unpatch_monai_swin_unetr() is True
    assert crumpet.unpatch_monai_swin_unetr() is False


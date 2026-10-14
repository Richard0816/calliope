"""Regression tests for the 2026-10-14 GUI walkthrough audit (docs/ux_audit).

Headless: no suite2p run, no GPU, no real recording.
"""
from __future__ import annotations

import pytest

from calliope.core import suite2p_pipeline as sp


def _cfg(tmp_path, monkeypatch, **kw):
    class _Sizer:
        def pick(self, _folder):
            return 777

    monkeypatch.setattr(sp.utils, "get_adaptive_sizer", lambda: _Sizer())
    monkeypatch.setattr(sp.utils, "change_nbinned_according_to_free_ram",
                        lambda _f: 5000)
    monkeypatch.setattr(sp.utils, "_peek_tiff_frame_shape", lambda _f: None)
    return sp.Suite2pPipelineConfig(tiff_folder=str(tmp_path),
                                    save_folder=str(tmp_path),
                                    verbose=False, **kw)


# --- finding 2: a user-set batch_size is no longer overwritten ---

@pytest.mark.parametrize("override, expected", [
    (None, 777),                                    # unset -> auto-pick
    ({"registration": {"batch_size": 200}}, 200),   # popout edit wins
])
def test_batch_size_override_respected(tmp_path, monkeypatch, override,
                                       expected):
    cfg = _cfg(tmp_path, monkeypatch, settings_override=override)
    _db, settings = sp.load_base_settings(cfg)
    assert sp.get_setting(settings, "batch_size") == expected

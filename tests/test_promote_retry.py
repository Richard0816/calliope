"""Tracker #33 item 10: a failed filtered-dF/F rewrite after Promote can be
retried without replacing the saved (hand-promoted) mask."""
from types import SimpleNamespace

from calliope.core import detection_run
from calliope.tabs.suite2p import tab as s2p_tab
from calliope.tabs.suite2p.curation_popout import CurationPopout


def _fake_tab(fail_times):
    calls = {"run": 0, "published": 0, "released": 0}

    def run(plane0):
        calls["run"] += 1
        if calls["run"] <= fail_times:
            raise PermissionError("file in use")

    state = SimpleNamespace(
        release_memmaps=lambda: calls.__setitem__("released", calls["released"] + 1),
        set_plane0=lambda p: calls.__setitem__("published", calls["published"] + 1))
    return SimpleNamespace(state=state, _run_filtered_dff=run,
                           _redraw_with_bg=lambda: None,
                           _filtered_matches_keep=lambda p: False), calls


def _setup(tmp_path, monkeypatch, answers):
    (tmp_path / "r0p7_dff.memmap.float32").write_bytes(b"")
    monkeypatch.setattr(detection_run, "has_filtered_outputs", lambda p: True)
    it = iter(answers)
    monkeypatch.setattr(s2p_tab.messagebox, "askretrycancel", lambda *a, **k: next(it))


def test_retry_rewrites_and_publishes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [True])
    fake, calls = _fake_tab(fail_times=1)
    s2p_tab.Suite2pTab._regenerate_after_promote(fake, tmp_path)
    assert calls["run"] == 2 and calls["published"] == 1
    assert calls["released"] == 2  # handles dropped again before the retry


def test_cancel_leaves_downstream_untouched(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [False])
    fake, calls = _fake_tab(fail_times=1)
    s2p_tab.Suite2pTab._regenerate_after_promote(fake, tmp_path)
    assert calls["run"] == 1 and calls["published"] == 0


def test_promote_with_nothing_flipped_retries_rewrite():
    seen = []
    pop = SimpleNamespace(_flipped=set(), _retry_rewrite=True,
                          _on_iscell_changed=lambda i, v: seen.append((i, v)))
    CurationPopout._on_promote(pop)
    assert seen == [(-1, -1)]
    pop._retry_rewrite = False          # standalone curation app
    CurationPopout._on_promote(pop)
    assert seen == [(-1, -1)]


# Tracker #33 item 13: an unchanged Promote must not rewrite (and so must not
# release the memmaps, which drops Tab 6's in-memory linkage).

def _plane0(tmp_path, keep):
    import numpy as np
    n, t = len(keep), 12
    np.save(tmp_path / "F.npy", np.ones((n, t), np.float32))
    np.save(tmp_path / "iscell.npy", np.c_[keep, keep].astype(float))
    np.save(tmp_path / "predicted_cell_mask.npy", np.asarray(keep, bool))
    np.memmap(tmp_path / "r0p7_dff.memmap.float32", np.float32, "w+",
              shape=(t, n))[:] = np.arange(t * n).reshape(t, n)
    detection_run.compute_filtered_dff(tmp_path)
    return tmp_path


def test_unchanged_mask_skips_rewrite(tmp_path):
    import numpy as np
    p = _plane0(tmp_path, [1, 0, 1, 1])
    assert s2p_tab.Suite2pTab._filtered_matches_keep(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 1, 1, 1], bool))
    assert not s2p_tab.Suite2pTab._filtered_matches_keep(p)


def test_reopened_popout_offers_retry_only_when_pending(tmp_path):
    import numpy as np
    from calliope.tabs.suite2p.curation_popout import _filtered_rewrite_pending
    p = _plane0(tmp_path, [1, 0, 1, 1])
    assert not _filtered_rewrite_pending(p)
    np.save(p / "predicted_cell_mask.npy", np.array([1, 1, 1, 1], bool))
    assert _filtered_rewrite_pending(p)


def test_truncated_filtered_file_is_rewritten(tmp_path):
    p = _plane0(tmp_path, [1, 0, 1, 1])
    with open(p / "r0p7_filtered_dff.memmap.float32", "r+b") as fh:
        fh.truncate(8)
    assert not s2p_tab.Suite2pTab._filtered_matches_keep(p)


def test_unchanged_promote_keeps_handles(tmp_path, monkeypatch):
    p = _plane0(tmp_path, [1, 0, 1, 1])
    fake, calls = _fake_tab(fail_times=0)
    fake._filtered_matches_keep = s2p_tab.Suite2pTab._filtered_matches_keep
    s2p_tab.Suite2pTab._regenerate_after_promote(fake, p)
    assert calls == {"run": 0, "published": 0, "released": 0}

"""Tracker #33 item 10: a failed filtered-dF/F rewrite after Promote can be
retried without replacing the saved (hand-promoted) mask."""
from types import SimpleNamespace

from calliope.core import detection_run, generation
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


def test_promote_with_nothing_flipped_confirms_then_resyncs(tmp_path, monkeypatch):
    import numpy as np
    from calliope.tabs.suite2p import curation_popout as cp
    p = _plane0(tmp_path, [1, 0, 1, 1])
    status, asked, answer = [], [], [False]
    monkeypatch.setattr(cp.messagebox, "askyesno",
                        lambda t, m: asked.append(m) or answer[0])
    seen = []
    pop = SimpleNamespace(
        _flipped=set(), _retry_rewrite=True, _plane0=p, _n_total=4,
        _train_status_var=SimpleNamespace(set=status.append),
        _on_iscell_changed=lambda i, v: seen.append((i, v)))
    CurationPopout._on_promote(pop)              # nothing drifted
    assert seen == [] and asked == [] and status
    np.save(p / "predicted_cell_mask.npy", np.array([1, 1, 1, 1], bool))
    CurationPopout._on_promote(pop)              # drift, user says No
    assert seen == [] and "keeps 3" in asked[0] and "now keeps 4" in asked[0]
    answer[0] = True
    CurationPopout._on_promote(pop)              # drift, user says Yes
    assert seen == [(-1, -1)]
    pop._retry_rewrite = False                   # standalone curation app
    CurationPopout._on_promote(pop)
    assert seen == [(-1, -1)] and len(asked) == 2


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


def test_stale_window_never_rewrites_current_recording(tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(s2p_tab.messagebox, "showwarning",
                        lambda *a, **k: warned.append(a))
    regen = []
    fake = SimpleNamespace(
        _panel_cache={"plane0": tmp_path / "current", "n_total": 4},
        _redraw_with_bg=lambda: None,
        _regenerate_after_promote=lambda p: regen.append(p))
    s2p_tab.Suite2pTab._on_iscell_flipped(fake, -1, -1, plane0=tmp_path / "old")
    assert regen == [] and len(warned) == 1


def test_matching_window_still_rewrites(tmp_path, monkeypatch):
    regen = []
    fake = SimpleNamespace(
        _panel_cache={"plane0": tmp_path, "n_total": 4},
        _redraw_with_bg=lambda: None,
        _regenerate_after_promote=lambda p: regen.append(p))
    monkeypatch.setattr(s2p_tab, "load_keep_mask", lambda p, n: None)
    s2p_tab.Suite2pTab._on_iscell_flipped(fake, -1, -1, plane0=tmp_path)
    assert regen == [tmp_path]


def test_rebuilt_dff_with_same_mask_is_rewritten(tmp_path):
    from calliope.tabs.suite2p.curation_popout import filtered_matches_keep
    p = _plane0(tmp_path, [1, 0, 1, 1])
    b = generation.new_dff_build(p)
    generation.finish_dff_build(p, b)
    assert not filtered_matches_keep(p)


def test_pending_stamp_is_rewritten(tmp_path):
    from calliope.tabs.suite2p.curation_popout import filtered_matches_keep
    p = _plane0(tmp_path, [1, 0, 1, 1])
    generation.begin_filtered(p)
    assert not filtered_matches_keep(p)


def test_callback_follows_repoint(tmp_path, monkeypatch):
    import functools
    monkeypatch.setattr(s2p_tab, "load_keep_mask", lambda p, n: None)
    monkeypatch.setattr(s2p_tab.messagebox, "showwarning", lambda *a, **k: None)
    scratch, final = tmp_path / "scratch", tmp_path / "final"
    regen = []
    popout = SimpleNamespace(_plane0=scratch, winfo_exists=lambda: True)
    fake = SimpleNamespace(
        _panel_cache={"plane0": scratch, "n_total": 4},
        _curation_popout=popout, _redraw_with_bg=lambda: None,
        _regenerate_after_promote=lambda p: regen.append(p))
    fake._on_iscell_flipped = functools.partial(
        s2p_tab.Suite2pTab._on_iscell_flipped, fake)
    cb = s2p_tab.Suite2pTab._curation_callback(fake, {"w": popout})
    s2p_tab.Suite2pTab._repoint_after_copy(fake, scratch, final)
    cb(-1, -1)
    assert regen == [final]

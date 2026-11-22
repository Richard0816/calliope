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
                           _redraw_with_bg=lambda: None), calls


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
    pop = SimpleNamespace(_flipped=set(),
                          _on_iscell_changed=lambda i, v: seen.append((i, v)))
    CurationPopout._on_promote(pop)
    assert seen == [(-1, -1)]

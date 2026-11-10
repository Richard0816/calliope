"""Suite2p detection must warn before it runs past the commit limit.

The 23 GB in vivo recording (1024 x 1024 x 10,923 frames) died in detection
(worker exit 0xC0000142) at the machine's 34 GB commit limit; the batch-size
sizing only looked at free RAM.
"""
import calliope.core.sparse_plus_cellpose as spc

SETTINGS = {"tau": 0.25, "fs": 15.17, "detection": {"nbins": 5000, "bin_size": None}}


def test_estimate_follows_suite2p_binning(monkeypatch):
    monkeypatch.setattr(spc, "available_commit_bytes", lambda: 2**50)
    # bin_size = max(1, 10923 // 5000, round(0.25 * 15.17)) = 4 -> 2730 binned frames
    need = spc.check_detection_memory(1024, 1024, 10923, SETTINGS, verbose=False)
    assert need == int(spc.DETECTION_MEM_FACTOR * 2730 * 1024 * 1024 * 4)


def test_warns_but_never_stops_when_commit_is_short(monkeypatch, capsys):
    # The factor is uncalibrated, so over the limit it warns (even when not
    # verbose) and the run goes on.
    monkeypatch.setattr(spc, "available_commit_bytes", lambda: 10 * 2**30)
    spc.check_detection_memory(1024, 1024, 10923, SETTINGS, verbose=False)
    out = capsys.readouterr().out
    assert "WARNING" in out and "can still be committed" in out
    assert "uncalibrated" in out and "may fail" in out


def test_warns_near_the_limit_and_passes_when_ample(monkeypatch, capsys):
    monkeypatch.setattr(spc, "available_commit_bytes", lambda: 2**50)
    need = spc.check_detection_memory(1024, 1024, 10923, SETTINGS, verbose=False)
    monkeypatch.setattr(spc, "available_commit_bytes", lambda: int(need / 0.9))
    spc.check_detection_memory(1024, 1024, 10923, SETTINGS)
    assert "WARNING" in capsys.readouterr().out
    monkeypatch.setattr(spc, "available_commit_bytes", lambda: 10 * need)
    spc.check_detection_memory(1024, 1024, 10923, SETTINGS)
    assert capsys.readouterr().out == ""

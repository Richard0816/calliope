"""Tests for the run manifest writer (core.export_manifest).

The key behaviour for run-reload: each stage's export must *accumulate*
its params into one manifest rather than overwrite, so a later stage's
export can't wipe the event-detection params a reload needs.
"""
from __future__ import annotations

import json
from pathlib import Path

from calliope.core.export_manifest import write_export_manifest


def _read(figs: Path) -> dict:
    return json.loads((figs / "manifest.json").read_text(encoding="utf-8"))


def test_first_write_records_params(tmp_path):
    figs = tmp_path / "calliope_figures"
    write_export_manifest(figs, rec_id="rec", params={"min_prominence": 0.3})
    man = _read(figs)
    assert man["rec_id"] == "rec"
    assert man["params"]["min_prominence"] == 0.3


def test_later_stage_export_does_not_clobber_event_params(tmp_path):
    figs = tmp_path / "calliope_figures"
    # Tab 5 exports event params...
    write_export_manifest(figs, rec_id="rec",
                          params={"min_prominence": 0.3, "mad_k": 5})
    # ...then Tab 7 (xcorr) exports with a totally different param set.
    write_export_manifest(figs, rec_id="rec",
                          params={"xcorr_max_lag_s": 2.0})
    man = _read(figs)
    # Event params survive the later export (the bug this fixes).
    assert man["params"]["min_prominence"] == 0.3
    assert man["params"]["mad_k"] == 5
    # And the later stage's params are present too.
    assert man["params"]["xcorr_max_lag_s"] == 2.0


def test_collision_later_call_wins(tmp_path):
    figs = tmp_path / "calliope_figures"
    write_export_manifest(figs, rec_id="rec", params={"shared": 1})
    write_export_manifest(figs, rec_id="rec", params={"shared": 2})
    assert _read(figs)["params"]["shared"] == 2


def test_prior_metadata_carried_forward_when_absent(tmp_path):
    figs = tmp_path / "calliope_figures"
    # First write seeds a gcamp_variant via params...
    write_export_manifest(figs, rec_id="rec",
                          params={"gcamp_variant": "jGCaMP8m"})
    # ...a later write that doesn't mention it must not drop it.
    write_export_manifest(figs, rec_id="rec", params={"min_prominence": 0.1})
    man = _read(figs)
    assert man["gcamp_variant"] == "jGCaMP8m"
    assert man["params"]["gcamp_variant"] == "jGCaMP8m"


def test_corrupt_existing_manifest_is_replaced_not_fatal(tmp_path):
    figs = tmp_path / "calliope_figures"
    figs.mkdir(parents=True)
    (figs / "manifest.json").write_text("{ broken", encoding="utf-8")
    write_export_manifest(figs, rec_id="rec", params={"min_prominence": 0.2})
    assert _read(figs)["params"]["min_prominence"] == 0.2

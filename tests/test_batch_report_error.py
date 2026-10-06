"""A failed stage must fill batch_report.csv's row-level ``error`` column.

``_fail_stage`` used to store the exception text only under
``stages[stage]["error"]``, which ``_write_report`` never writes, so a
failed row showed status=failed with a blank error cell.
"""

from __future__ import annotations

import csv
import threading
import time
import types

from calliope.tabs.batch.tab import BatchTab


def _stub():
    s = types.SimpleNamespace()
    s._current_row = None
    s._abort_flag = threading.Event()
    s._MAX_OOM_RETRIES = BatchTab._MAX_OOM_RETRIES
    s._is_oom_error = lambda msg: False
    s._stage_t0 = time.time()
    s._row_results = {"recording_id": "rec1", "status": "running",
                      "plane0": "", "total_s": 1.0, "error": None,
                      "stages": {}}
    s.logs = []
    s._append_log = s.logs.append
    s._on_row_done = lambda: None
    return s


def test_failed_stage_error_reaches_report(tmp_path):
    s = _stub()
    BatchTab._fail_stage(s, "lowpass", RuntimeError("bad cutoff"))
    assert s._row_results["error"] == "lowpass: bad cutoff"

    BatchTab._write_report(tmp_path, [s._row_results])
    with (tmp_path / "batch_report.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "lowpass: bad cutoff"
    assert rows[0]["lowpass_status"] == "failed"


def test_existing_row_error_is_kept():
    s = _stub()
    s._row_results["error"] = "earlier problem"
    BatchTab._fail_stage(s, "events", RuntimeError("x"))
    assert s._row_results["error"] == "earlier problem"

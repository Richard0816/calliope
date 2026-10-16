"""Tab 5's summary workbook write runs off the GUI thread (it froze the GUI
for ~23 s on a 679-ROI recording), in one open, behind a file lock."""
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd

from calliope.core import summary_writer as sw
from calliope.core.event_detection_run import write_event_summary_offload


def _data():
    return {
        "fps": 10.0, "T": 100, "N_kept": 2,
        "event_windows": [(1.0, 2.0)],
        "onsets_by_roi": [np.array([1.5]), np.array([])],
        "event_monotonicity": [{"event_id": 0, "n_active": 1}],
    }


def test_offload_writes_all_sheets_and_keeps_others(tmp_path):
    sw.write_sheet(tmp_path, "ROIs", {"roi": [0, 1]})
    path = write_event_summary_offload(str(tmp_path), _data())
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"ROIs", "Recording", "EventWindows", "EventOnsets",
            "RoiEventTimes", "EventMonotonicity"} <= set(sheets)
    rec = dict(zip(sheets["Recording"]["key"], sheets["Recording"]["value"]))
    assert float(rec["fps"]) == 10.0


def test_concurrent_writers_keep_each_others_sheets(tmp_path):
    sw.write_sheet(tmp_path, "Seed", {"x": [0]})

    def writer(name):
        for i in range(3):
            sw.write_sheet(tmp_path, name, {"i": [i]})

    ts = [threading.Thread(target=writer, args=(n,)) for n in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert {"Seed", "A", "B"} <= set(
        pd.read_excel(sw.summary_path(tmp_path), sheet_name=None))


def test_batch_publish_waits_for_summary_write():
    from calliope.tabs.event_detection.tab import EventDetectionTab
    published = []
    fake = SimpleNamespace(_suppress_manifest_write=True, _plane0=None,
                           _after_summary_write=lambda: published.append(1))
    EventDetectionTab._on_summary_written(fake, "path")
    assert published == [1] and fake._after_summary_write is None
    # A failed write still releases the batch.
    fake._after_summary_write = lambda: published.append(2)
    fake._on_summary_written = lambda p: EventDetectionTab._on_summary_written(fake, p)
    EventDetectionTab._on_summary_failed(fake, "Traceback ...")
    assert published == [1, 2]

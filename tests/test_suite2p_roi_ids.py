"""Every ROI id a user sees is the Suite2p id, never a position in a tab's
kept list: the Events sheets and the Tab 6 cluster popout."""
import numpy as np
import pandas as pd
import pytest

from calliope.core import summary_writer as sw


def test_event_sheets_use_suite2p_ids(tmp_path):
    # Tab 5 kept Suite2p ROIs 4, 17 and 30 (positions 0, 1, 2).
    sw.write_events_sheets(
        tmp_path, event_windows=[(1.0, 2.0)],
        onsets_by_roi=[np.array([1.5]), np.array([]), np.array([1.2, 9.0])],
        fps=10.0, roi_ids=[4, 17, 30])
    sheets = pd.read_excel(sw.summary_path(tmp_path), sheet_name=None)
    assert sorted(sheets["EventOnsets"]["roi"]) == [4, 30]
    assert list(sheets["RoiEventTimes"].columns) == ["roi_4", "roi_17", "roi_30"]
    with pytest.raises(ValueError):
        sw.write_events_sheets(tmp_path, [(1.0, 2.0)], [np.array([1.5])],
                               roi_ids=[4, 17])


def test_cluster_popout_suite2p_ids_and_median_trace(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    from calliope.tabs.clustering.cluster_popout import ClusterPopout
    try:
        T = 50
        dff = np.zeros((T, 4), dtype=np.float32)
        dff[:, 1] = np.linspace(0, 1, T)
        dff[:, 3] = np.linspace(0, 3, T)
        pop = ClusterPopout(root, tmp_path)
        pop.show_cluster(clicked_idx=3, cluster_label=2, dff=dff,
                         labels=np.array([1, 2, 1, 2]), stat=None, fps=10.0,
                         roi_ids=[5, 12, 40, 77], cluster_name="C2")
        status = pop._status_var.get()
        assert "C2" in status and "clicked ROI 77" in status
        assert "Suite2p ROIs: 12, 77" in status
        assert [t.get_text() for t in pop._ax_hm.get_yticklabels()] == ["12", "77"]
        median = pop._ax_md.get_lines()[0].get_ydata()
        np.testing.assert_allclose(median, np.median(dff[:, [1, 3]], axis=1))
        pop._on_close()
    finally:
        root.destroy()

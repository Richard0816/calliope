"""A sheet over Excel's size limit goes to a CSV beside the workbook instead of
failing the whole summary write (EventOnsets hit 1.34M rows on an in vivo
recording with ~7,600 ROIs)."""
import pandas as pd

from calliope.core import summary_writer as sw


def test_oversized_sheet_spills_to_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "_XLSX_MAX_ROWS", 5)
    big = pd.DataFrame({"event_id": range(10), "roi": range(10)})
    path = sw.write_sheets(tmp_path, {"EventOnsets": big, "EventWindows": big.head(2)})

    pd.testing.assert_frame_equal(pd.read_csv(path.with_name(f"{path.stem}_EventOnsets.csv")), big)
    note = pd.read_excel(path, sheet_name="EventOnsets")
    assert "10 rows" in note.loc[0, "note"]
    assert len(pd.read_excel(path, sheet_name="EventWindows")) == 2

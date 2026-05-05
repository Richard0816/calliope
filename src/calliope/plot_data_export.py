"""plot_data_export.py - dump the underlying data of a matplotlib Figure
to CSV via a tk save dialog.

Called by ``gui_common.attach_fig_toolbar`` when the user clicks the
"Save data..." button next to the navigation toolbar. Walks every Axes,
extracts Line2D xy data and AxesImage arrays, and writes the result as
either a single CSV (one Line2D / image) or a multi-sheet bundle
(``<basename>__line<i>.csv`` / ``<basename>__image<i>.csv``).

Pure-numpy + tkinter; no extra dependencies.
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np


def save_figure_data(figure, parent, basename: str = "plot_data") -> None:
    """Prompt for an output directory and write CSVs of every Line2D and
    AxesImage in ``figure``.

    Each Line2D becomes ``<basename>__ax<i>_line<j>.csv`` with columns
    ``x,y``. Each AxesImage becomes ``<basename>__ax<i>_image<j>.csv`` as
    a 2-D array (no header). If nothing exportable is found, shows an
    info dialog and returns.
    """
    out_dir = filedialog.askdirectory(
        parent=parent, title="Save figure data to folder",
    )
    if not out_dir:
        return
    out_root = Path(out_dir)
    written: list[Path] = []
    try:
        for i, ax in enumerate(figure.get_axes()):
            for j, line in enumerate(ax.get_lines()):
                xy = line.get_xydata()
                if xy.size == 0:
                    continue
                p = out_root / f"{basename}__ax{i}_line{j}.csv"
                np.savetxt(p, xy, delimiter=",", header="x,y", comments="")
                written.append(p)
            for j, im in enumerate(getattr(ax, "images", []) or []):
                arr = np.asarray(im.get_array())
                if arr.size == 0:
                    continue
                p = out_root / f"{basename}__ax{i}_image{j}.csv"
                if arr.ndim == 3:
                    # Flatten RGB(A) to columns of "channel<k>" per pixel.
                    h, w, c = arr.shape
                    flat = arr.reshape(h * w, c)
                    hdr = ",".join(f"channel{k}" for k in range(c))
                    np.savetxt(p, flat, delimiter=",", header=hdr, comments="")
                else:
                    np.savetxt(p, arr, delimiter=",")
                written.append(p)
    except Exception as e:
        messagebox.showerror(
            "Save data failed",
            f"Could not write figure data: {e}",
            parent=parent,
        )
        return

    if not written:
        messagebox.showinfo(
            "No data",
            "No Line2D or image data found on the figure.",
            parent=parent,
        )
        return

    messagebox.showinfo(
        "Saved",
        f"Wrote {len(written)} file(s) to:\n{out_root}",
        parent=parent,
    )

"""plot_data_export.py - dump the underlying data of a matplotlib Figure
to CSV via a tk save dialog.

Why this exists
---------------
Every figure in CalLIOPE has a "Save figure" button that exports a PNG.
That's enough for slides, but useless if you want to re-plot the data
in R / Prism / Excel later. This module walks the actual ``Line2D`` and
``AxesImage`` objects backing the figure and writes their numerics out
as CSV files -- one file per line, one file per image -- alongside the
PNG.

Called by ``gui_common.attach_fig_toolbar`` when the user clicks the
"Save data..." button next to the navigation toolbar.

Single CSV or several:
    - One Line2D / image  -> single ``<basename>__ax0_line0.csv``.
    - Multiple lines / images -> a bundle ``<basename>__ax{i}_line{j}.csv``
      / ``<basename>__ax{i}_image{j}.csv``.

Pure NumPy + tkinter; no extra dependencies, no pandas.
"""

from __future__ import annotations

# ``pathlib.Path`` is a more convenient cross-platform alternative to
# bare strings or ``os.path`` -- it overrides ``/`` so ``out_root /
# "foo.csv"`` is path joining, not division. Roughly the same idea as
# R's ``fs::path``.
from pathlib import Path

# ``filedialog.askdirectory`` pops up the OS folder picker and returns
# the chosen path (or "" if the user cancelled).
# ``messagebox.showerror`` / ``showinfo`` are the standard alert pop-ups.
from tkinter import filedialog, messagebox

import numpy as np


def save_figure_data(figure, parent, basename: str = "plot_data") -> None:
    """Prompt for an output directory and write CSVs of every Line2D and
    AxesImage in ``figure``.

    Each Line2D becomes ``<basename>__ax<i>_line<j>.csv`` with columns
    ``x,y``. Each AxesImage becomes ``<basename>__ax<i>_image<j>.csv`` as
    a 2-D array (no header). If nothing exportable is found, shows an
    info dialog and returns.

    ``parent`` is the Tk widget the dialog should be parented to (used
    to pin the modal dialog to the right window). ``basename`` is the
    leading filename stem -- often the figure's purpose, like
    ``"event_heatmap"``.
    """
    # askdirectory returns the empty string if the user hits Cancel.
    out_dir = filedialog.askdirectory(
        parent=parent, title="Save figure data to folder",
    )
    if not out_dir:
        return
    out_root = Path(out_dir)
    # ``written`` accumulates the paths we successfully wrote so we can
    # report them in the success dialog at the end.
    written: list[Path] = []
    try:
        # ``enumerate(seq)`` yields ``(index, item)`` pairs. Equivalent
        # to ``for (i in seq_along(seq))`` plus ``seq[i]`` in R.
        for i, ax in enumerate(figure.get_axes()):
            # --- Line plots ---
            for j, line in enumerate(ax.get_lines()):
                # ``get_xydata`` returns an (N, 2) array of (x, y).
                xy = line.get_xydata()
                if xy.size == 0:
                    continue
                p = out_root / f"{basename}__ax{i}_line{j}.csv"
                # ``np.savetxt`` writes a 2-D array as plain CSV.
                # ``comments=""`` prevents numpy prefixing the header
                # line with a ``# ``.
                np.savetxt(p, xy, delimiter=",", header="x,y", comments="")
                written.append(p)
            # --- Bitmap images / heatmaps ---
            # ``getattr(ax, "images", []) or []`` is defensive: if
            # ``ax.images`` is missing (older matplotlib) or None, fall
            # back to an empty list so the loop below doesn't crash.
            for j, im in enumerate(getattr(ax, "images", []) or []):
                arr = np.asarray(im.get_array())
                if arr.size == 0:
                    continue
                p = out_root / f"{basename}__ax{i}_image{j}.csv"
                if arr.ndim == 3:
                    # RGB(A) image: flatten H x W x C into one row per
                    # pixel, with one column per channel. Lossless, but
                    # column meaning is "channel0/1/2/3" not "x/y".
                    h, w, c = arr.shape
                    flat = arr.reshape(h * w, c)
                    # Generator expression inside ``join``: comma-
                    # separated channel header.
                    hdr = ",".join(f"channel{k}" for k in range(c))
                    np.savetxt(p, flat, delimiter=",", header=hdr, comments="")
                else:
                    # 2-D heatmap: just write the matrix as-is.
                    np.savetxt(p, arr, delimiter=",")
                written.append(p)
    except Exception as e:
        # If anything goes wrong (permissions, disk full, weird
        # array shape) tell the user instead of crashing the GUI.
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


def save_columns_csv(parent, columns: dict, default_basename: str = "plot_data") -> None:
    """Variant used by tabs that want to dump *named columns* of
    different lengths (e.g. the cross-correlation violin window's
    per-pair distributions). matplotlib's ``Line2D`` / ``AxesImage``
    machinery doesn't recover those, so the panel hands us the dict
    directly.

    ``columns`` maps column name -> 1D numpy-like array. Columns can
    have different lengths; we pad with empty strings on the right.
    """
    # Single Save-As dialog because the output is one CSV (wide form).
    out_path = filedialog.asksaveasfilename(
        parent=parent,
        title="Save columns to CSV",
        defaultextension=".csv",
        initialfile=f"{default_basename}.csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not out_path:
        return
    try:
        names = list(columns.keys())
        if not names:
            messagebox.showinfo("No data", "Nothing to save.", parent=parent)
            return
        # Pad shorter columns with empty strings so the CSV stays
        # rectangular.
        max_len = max((len(np.asarray(v)) for v in columns.values()),
                      default=0)
        # ``with open(...) as fh:`` is Python's "use this file inside
        # this block, then close it automatically". The ``newline=""``
        # is important on Windows so csv doesn't double-up the line
        # endings.
        import csv
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            arrs = [np.asarray(columns[n]) for n in names]
            for row_i in range(max_len):
                row = []
                for arr in arrs:
                    if row_i < arr.size:
                        row.append(arr[row_i])
                    else:
                        row.append("")
                w.writerow(row)
        messagebox.showinfo("Saved",
                            f"Wrote columns to:\n{out_path}",
                            parent=parent)
    except Exception as e:
        messagebox.showerror(
            "Save data failed",
            f"Could not write CSV: {e}",
            parent=parent,
        )

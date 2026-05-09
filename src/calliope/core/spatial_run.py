"""Headless per-event spatial-propagation figure renderer.

Why this module exists
----------------------
Tab 8 is purely visual -- no computation to run headlessly -- but
the batch runner still wants the per-event activation-order PNGs
saved alongside everything else under ``calliope_figures/``. This
module renders one PNG per event using the same
``order_map_for_event`` + ``paint_order_map`` pipeline the tab uses,
without any Tk dependency.

Public surface
--------------
``render_spatial_event_figures(plane0, event_data, *, figures_dir)``
    Save one PNG per event window. ``event_data`` is the dict
    returned by :func:`event_detection_run.run_event_detection`
    (must contain ``event_windows``, ``A``, ``first_time``,
    ``kept_idx``, ``fps``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import scale as scale_mod
from . import spatial as spatial_helpers
from . import utils


def render_spatial_event_figures(
    plane0: Path,
    event_data: dict,
    *,
    figures_dir: Path,
) -> list[str]:
    """Render the order-map-with-arrows panel (Tab 8's most
    informative single view) for every event window. Saved as
    ``event_001.png``, ``event_002.png``, ... under ``figures_dir``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plane0 = Path(plane0)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ev = np.asarray(event_data.get("event_windows"))
    if ev is None or ev.size == 0:
        return []
    A = np.asarray(event_data["A"])
    first_time = np.asarray(event_data["first_time"])
    kept_idx = np.asarray(event_data["kept_idx"])
    fps: Optional[float] = event_data.get("fps")

    view = utils.load_plane_view(plane0)
    Ly, Lx = int(view["Ly"]), int(view["Lx"])
    pix_to_um = scale_mod.resolve_pix_to_um(view)
    stat_all = np.load(plane0 / "stat.npy", allow_pickle=True)
    stat_filtered = [stat_all[i] for i in kept_idx.tolist()]

    if pix_to_um is not None:
        scale = float(pix_to_um)
        extent = [0, Lx * scale, 0, Ly * scale]
        xlabel, ylabel = "X (µm)", "Y (µm)"
    else:
        scale = 1.0
        extent = None
        xlabel, ylabel = "X (px)", "Y (px)"

    written: list[str] = []
    n_events = ev.shape[0]
    for ev_idx in range(n_events):
        first_col = first_time[:, ev_idx]
        active_col = A[:, ev_idx]
        order_rank = spatial_helpers.order_map_for_event(
            first_col, active_col)
        img = spatial_helpers.paint_order_map(
            order_rank, stat_filtered, Ly, Lx)
        t0 = float(ev[ev_idx, 0]); t1 = float(ev[ev_idx, 1])
        n_active = int(active_col.sum())
        n_total = int(active_col.size)
        frac = n_active / n_total if n_total else 0.0

        frame_groups = []
        if fps:
            frame_groups = spatial_helpers.event_frame_centroids(
                first_col, active_col, stat_filtered, float(fps))
        cxs = np.array([g["cx"] * scale for g in frame_groups])
        cys = np.array([g["cy"] * scale for g in frame_groups])

        fig = plt.Figure(figsize=(7, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            img, origin="lower",
            cmap=spatial_helpers.CYAN_TO_RED,
            aspect="equal", vmin=0.0, vmax=1.0, extent=extent,
        )
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(
            f"Event {ev_idx + 1} / {n_events}  ({t0:.2f}-{t1:.2f} s)\n"
            f"active = {n_active} / {n_total} ({100 * frac:.1f}%)",
            fontsize=10,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                            ticks=[0.0, 0.5, 1.0])
        cbar.ax.set_yticklabels(["earliest", "", "latest"])

        if frame_groups:
            ax.scatter(cxs, cys, s=18, c="white",
                       edgecolors="black", linewidths=0.6, zorder=4)
            for i in range(len(frame_groups) - 1):
                x0, y0 = cxs[i], cys[i]
                x1, y1 = cxs[i + 1], cys[i + 1]
                if abs(x1 - x0) + abs(y1 - y0) < 1e-9:
                    continue
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->", color="white",
                        lw=1.6, shrinkA=2, shrinkB=2,
                        mutation_scale=14),
                    zorder=5,
                )

        out = figures_dir / f"event_{ev_idx + 1:03d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(str(out))

    return written

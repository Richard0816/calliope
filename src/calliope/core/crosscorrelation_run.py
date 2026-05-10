"""Headless cluster x cluster cross-correlation for batch execution.

Why this module exists
----------------------
The maths in :mod:`calliope.core.crosscorrelation` is already
callable headlessly -- ``run_cluster_xcorr_full_fast`` /
``run_cluster_xcorr_per_event_fast`` write per-pair summary CSVs.
What was missing for batch use is a thin wrapper that drives both
runs end-to-end (full + per-event when event windows are available),
loads the resulting CSVs, and renders the same violin plots Tab 7
shows in the GUI.

Public surface
--------------
``run_crosscorrelation(plane0, params, *, event_windows=None,
                       figures_dir=None)``
    Run the full-recording cross-correlation, then per-event if
    ``event_windows`` is supplied. If ``figures_dir`` is set, save
    zero-lag-correlation and best-lag violin plots for each output
    folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from . import crosscorrelation as xc


def _save_violin_pair(pair_data, *, save_path: Path, title_suffix: str):
    """Render the same two-violin layout the Tab 7 GUI shows
    (zero-lag correlation + best lag) and save to ``save_path``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from ..tabs.crosscorrelation.tab import (
        _get_metric_arrays, _plot_violin, _plot_lag_violin,
    )

    corr_labels, corr_arrays = _get_metric_arrays(pair_data, "zero_lag_corr")
    lag_labels, lag_arrays = _get_metric_arrays(pair_data, "best_lag_sec")

    fig, (ax_corr, ax_lag) = plt.subplots(2, 1, figsize=(10, 8),
                                          tight_layout=True)
    if corr_arrays:
        _plot_violin(
            ax_corr, corr_labels, corr_arrays,
            "Zero-lag correlation",
            f"Per-pair zero-lag correlation  ({title_suffix})",
            show_sig=False,
        )
    else:
        ax_corr.set_axis_off()
        ax_corr.text(0.5, 0.5, "No pair data", ha="center", va="center",
                     transform=ax_corr.transAxes)

    if lag_arrays:
        _plot_lag_violin(
            ax_lag, lag_labels, lag_arrays,
            "Lag at peak correlation (s)",
            f"Per-pair best-lag distribution  ({title_suffix})",
        )
    else:
        ax_lag.set_axis_off()
        ax_lag.text(0.5, 0.5, "No pair data", ha="center", va="center",
                    transform=ax_lag.transAxes)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_crosscorrelation(
    plane0: Path,
    params: dict,
    *,
    event_windows: Optional[Sequence] = None,
    figures_dir: Optional[Path] = None,
    progress_cb=None,
) -> dict:
    """Drive ``run_cluster_xcorr_full_fast`` and (optionally)
    ``run_cluster_xcorr_per_event_fast`` and save violin figures.

    Parameters
    ----------
    plane0 : Path
        Suite2p plane folder.
    params : dict
        ``prefix`` (default ``"r0p7_filtered_"``),
        ``cluster_folder`` (default ``""`` -- ROI files live directly
        under ``<prefix>cluster_results/``; pass a name to use an
        extra subfolder under there),
        ``fps`` (must be supplied; usually carried over from
        upstream stages),
        ``max_lag_seconds`` (default 2.0),
        ``zero_lag`` (default True),
        ``use_gpu`` (default True).
    event_windows : iterable of (start_s, end_s) or None
        If supplied, also run the per-event xcorr.
    figures_dir : Path or None
        If set, save ``full.png`` and ``per_event.png`` violin plots.

    Returns
    -------
    dict
        ``{full_outdir, per_event_outdir}`` (paths or None).
    """
    plane0 = Path(plane0)
    prefix = str(params.get("prefix", "r0p7_filtered_"))
    cluster_folder = str(params.get("cluster_folder", ""))
    fps = float(params.get("fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError("run_crosscorrelation requires positive fps in params")
    max_lag = float(params.get("max_lag_seconds", 2.0))
    zero_lag = bool(params.get("zero_lag", True))
    use_gpu = bool(params.get("use_gpu", True))

    full_subdir = "cross_correlation_full"
    xc.run_cluster_xcorr_full_fast(
        plane0, prefix=prefix, fps=fps, cluster_folder=cluster_folder,
        max_lag_seconds=max_lag, zero_lag=zero_lag, use_gpu=use_gpu,
        output_subdir=full_subdir, progress_cb=progress_cb,
    )
    full_outdir = (plane0 / f"{prefix}cluster_results" / cluster_folder
                   / full_subdir)

    per_event_outdir: Optional[Path] = None
    if event_windows is not None and len(event_windows) > 0:
        per_event_subdir = "cross_correlation_per_event"
        xc.run_cluster_xcorr_per_event_fast(
            plane0, prefix=prefix, fps=fps,
            cluster_folder=cluster_folder, event_windows=event_windows,
            max_lag_seconds=max_lag, zero_lag=zero_lag, use_gpu=use_gpu,
            output_subdir=per_event_subdir, progress_cb=progress_cb,
        )
        per_event_outdir = (plane0 / f"{prefix}cluster_results"
                            / cluster_folder / per_event_subdir)

    if figures_dir is not None:
        from ..tabs.crosscorrelation.tab import _load_pair_data
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        if full_outdir.exists():
            pair_data = _load_pair_data(full_outdir)
            _save_violin_pair(
                pair_data, save_path=figures_dir / "full.png",
                title_suffix="full recording",
            )
        if per_event_outdir is not None and per_event_outdir.exists():
            pair_data = _load_pair_data(per_event_outdir)
            _save_violin_pair(
                pair_data, save_path=figures_dir / "per_event.png",
                title_suffix="per event",
            )

    return {
        "full_outdir": str(full_outdir) if full_outdir.exists() else None,
        "per_event_outdir": (str(per_event_outdir)
                             if per_event_outdir is not None
                             and per_event_outdir.exists() else None),
    }

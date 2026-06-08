"""Micrometre scale-bar helpers for spatial figures.

Kept separate from :mod:`calliope.core.scale` (which is matplotlib-free and
used on headless resolve paths) so importing the pixel<->µm *math* never
drags matplotlib in. Everything here is render-time only.

The single entry point is :func:`add_scale_bar`, used by every spatial
figure (Tab 3 detection panels, Tab 6 clustering maps, Tab 8 event-order
maps). It draws a labelled bar sized in micrometres using matplotlib's
``AnchoredSizeBar`` (ships with matplotlib -- no extra dependency) and is a
no-op when no pixel calibration is available, so callers can wire it in
unconditionally and keep the pixel fallback for free.
"""

from __future__ import annotations

import math
from typing import Optional

from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def nice_bar_length_um(span_um: float, target_frac: float = 0.2
                       ) -> Optional[float]:
    """Pick a human-friendly 1/2/5x10^n scale-bar length.

    ``span_um`` is the full field-of-view width in micrometres; the bar
    targets ``target_frac`` of it (default 20%) and snaps to the nearest
    1, 2, or 5 times a power of ten so the label reads as a round number
    (e.g. a 3081 µm FOV -> 500 µm, a 120 µm FOV -> 20 µm).

    Returns ``None`` for a non-positive span (nothing sensible to draw).
    """
    if span_um is None or span_um <= 0:
        return None
    raw = float(span_um) * float(target_frac)
    if raw <= 0:
        return None
    exp = math.floor(math.log10(raw))
    base = raw / (10.0 ** exp)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0 ** exp)


def add_scale_bar(ax, pix_to_um: Optional[float], *,
                  axes_in_um: bool,
                  span_um: Optional[float] = None,
                  length_um: Optional[float] = None,
                  loc: str = "lower right",
                  color: str = "white") -> Optional[float]:
    """Draw a micrometre scale bar on ``ax``; return the bar length (µm).

    No-op (returns ``None``) when ``pix_to_um`` is missing/non-positive, so
    callers can invoke it unconditionally and fall back to bare pixels.

    Parameters
    ----------
    pix_to_um
        Micrometres per pixel for the recording (``None`` -> skip).
    axes_in_um
        ``True`` when the axes' data coordinates are already micrometres
        (an ``imshow`` ``extent`` in µm); the bar ``size`` is then the
        length in µm directly. ``False`` when the axes are still in pixel
        data coordinates (e.g. an ``axis('off')`` image with no extent);
        the bar ``size`` is converted to pixels as ``length_um / pix_to_um``.
    span_um
        Field-of-view width in µm, used to auto-pick a round ``length_um``
        via :func:`nice_bar_length_um` when ``length_um`` isn't given.
    length_um
        Explicit bar length in µm (overrides ``span_um``).
    """
    if pix_to_um is None or pix_to_um <= 0:
        return None
    if length_um is None:
        if span_um is None:
            return None
        length_um = nice_bar_length_um(span_um)
    if not length_um or length_um <= 0:
        return None

    # ``size`` is expressed in the axes' data units.
    size = length_um if axes_in_um else length_um / pix_to_um
    bar = AnchoredSizeBar(
        ax.transData, size, f"{length_um:g} µm", loc,
        pad=0.4, borderpad=0.5, sep=4, frameon=False, color=color,
        size_vertical=size * 0.03,
        fontproperties=FontProperties(size=8),
    )
    ax.add_artist(bar)
    return float(length_um)

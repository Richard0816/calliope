"""Unit tests for ``refine_onsets`` (per-ROI onset backtrack + optional gate).

These drive the *real* detection chain -- lowpass_causal_1d -> sg_first_-
derivative_1d -> mad_z -> hysteresis_onsets -> refine_onsets -- rather than
feeding fabricated indices, so they exercise onsets the way the pipeline
actually produces them (an earlier version fed arbitrary indices and passed
for reasons unrelated to what it claimed to test).
"""
import numpy as np

from calliope.core.utils import (refine_onsets, lowpass_causal_1d,
                                  sg_first_derivative_1d, mad_z,
                                  hysteresis_onsets)

FPS = 15.07
CUT, ORDER, SGWIN, SGPOLY = 1.0, 2, 333, 2   # production low-pass / SG defaults
Z_HI, Z_LO, MIN_SEP = 3.5, 1.5, 0.1
T = int(300.0 * FPS)


def _ca(T, t0_frame, amp, fps, tau_r=0.20, tau_d=1.2):
    t = (np.arange(T) - t0_frame) / fps
    e = np.zeros(T)
    m = t >= 0
    e[m] = amp * (1 - np.exp(-t[m] / tau_r)) * np.exp(-t[m] / tau_d)
    return e


def _pipeline_onsets(raw):
    """raw dF/F -> (lp, hysteresis onset frames) exactly as detection does."""
    lp, _, _ = lowpass_causal_1d(raw, fps=FPS, cutoff_hz=CUT, order=ORDER)
    lp = np.asarray(lp, float)
    dt = sg_first_derivative_1d(lp, fps=FPS, win_ms=SGWIN, poly=SGPOLY)
    z = mad_z(np.asarray(dt, float))[0]
    return lp, hysteresis_onsets(z, Z_HI, Z_LO, FPS, min_sep_s=MIN_SEP)


def test_refine_onsets_reexported_by_tab_logic():
    """run_event_detection calls it as ``ed_utils.refine_onsets`` (the tab
    logic shim), so it must be re-exported there, not only in core.utils."""
    from calliope.tabs.event_detection import logic
    assert logic.refine_onsets is refine_onsets


def test_resolve_backtrack_auto():
    """0 -> auto = max(tau, 0.8/cutoff); a positive value is an override."""
    from pathlib import Path
    from calliope.core.event_detection_run import _resolve_backtrack_s
    nop = Path("does_not_exist_plane0")  # tau unreadable -> 0.0
    assert _resolve_backtrack_s(0.6, nop, {}) == 0.6               # override
    assert _resolve_backtrack_s(0.0, nop, {"cutoff_hz": 1.0}) == 0.8  # lowpass floor
    assert abs(_resolve_backtrack_s(0.0, nop, {"cutoff_hz": 2.0}) - 0.4) < 1e-9


def test_backtracks_crossing_to_rise_foot():
    """A real event's hysteresis crossing is relocated to at/before the
    rising flank -- never after the crossing, never past the peak."""
    rng = np.random.default_rng(0)
    f0 = 1500
    raw = rng.normal(0.0, 0.05, T) + _ca(T, f0, 1.2, FPS)
    lp, onsets = _pipeline_onsets(raw)
    assert onsets.size >= 1
    crossing = int(onsets[np.argmin(np.abs(onsets - f0))])

    out = refine_onsets(onsets, lp, FPS, min_rise_dff=0.0, backtrack_s=1.5)
    foot = int(out[np.argmin(np.abs(out - f0))])
    peak = f0 + int(np.argmax(lp[f0:f0 + int(2 * FPS)]))
    assert foot <= crossing            # backtracked, never later
    assert foot < peak                 # sits on the rise, not the peak
    assert abs(foot - f0) <= int(1.5 * FPS)   # near the true onset


def test_absolute_floor_drops_noise_keeps_real():
    """Through the real chain: an absolute dF/F floor removes noise-amplitude
    onsets while retaining a genuine transient. (A per-ROI robust-z floor
    cannot do this -- see module docstring.)"""
    rng = np.random.default_rng(1)
    # one real event amid baseline noise
    raw = rng.normal(0.0, 0.08, T) + _ca(T, 1500, 1.5, FPS)
    lp, onsets = _pipeline_onsets(raw)
    assert onsets.size >= 2            # the event + some noise crossings

    kept0 = refine_onsets(onsets, lp, FPS, min_rise_dff=0.0)
    kept = refine_onsets(onsets, lp, FPS, min_rise_dff=0.25)

    assert kept.size < kept0.size      # the floor drops some noise onsets
    # the real event survives the floor
    assert np.any(np.abs(kept - 1500) <= int(1.5 * FPS))


def test_floor_zeros_out_a_flat_noise_roi():
    """A pure-noise (phantom) ROI loses essentially all onsets under the
    absolute floor, while the floor=0 path keeps the backtracked noise onsets."""
    rng = np.random.default_rng(2)
    raw = rng.normal(0.0, 0.08, T)     # no events at all
    lp, onsets = _pipeline_onsets(raw)
    kept0 = refine_onsets(onsets, lp, FPS, min_rise_dff=0.0)
    kept = refine_onsets(onsets, lp, FPS, min_rise_dff=0.25)
    assert kept.size <= max(1, kept0.size // 4)


def test_monotonic_in_floor():
    """Higher floor never keeps more onsets."""
    rng = np.random.default_rng(3)
    raw = rng.normal(0.0, 0.08, T) + _ca(T, 1500, 0.9, FPS)
    lp, onsets = _pipeline_onsets(raw)
    sizes = [refine_onsets(onsets, lp, FPS, min_rise_dff=f).size
             for f in (0.0, 0.1, 0.25, 0.5, 1.0)]
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))


def test_edges_empty_and_dedupe():
    rng = np.random.default_rng(4)
    lp = rng.normal(0.0, 0.02, T)
    assert refine_onsets(np.array([], dtype=int), lp, FPS).size == 0
    # out-of-range index is clamped, not an IndexError
    assert refine_onsets(np.array([T + 5]), lp, FPS, backtrack_s=1.5).size <= 1
    # two crossings on one rise, sharing a single local-min foot, collapse to
    # one. A unique minimum at frame 1490 (descend into it, ramp out of it)
    # lies within both crossings' backtrack windows, so both walk back to it.
    ramp = np.empty(T)
    ramp[:1490] = np.linspace(0.1, 0.001, 1490)  # descends toward the min
    ramp[1490:1510] = np.linspace(0.0, 2.0, 20)  # unique min at 1490, rises out
    ramp[1510:] = 2.0
    # two crossings on the rise, both within the backtrack window of the foot
    out = refine_onsets(np.array([1493, 1496]), ramp, FPS, min_rise_dff=0.0)
    assert out.size == 1 and int(out[0]) == 1490


def test_foot_not_dragged_onto_flat_baseline():
    """On a flat baseline the foot is the trough the rise climbs out of, not a
    deeper noise dip earlier in the backtrack window. (The old argmin-over-
    window foot jumped to that dip -- up to ``backtrack_s`` early, ~1 s on
    real data -- dragging population event windows early too.)"""
    lp = np.zeros(T)
    lp[1495] = -0.05                              # noise dip inside the window
    lp[1500:1520] = np.linspace(0.0, 2.0, 20)     # rise starts at 1500
    lp[1520:] = 2.0
    out = refine_onsets(np.array([1505]), lp, FPS, backtrack_s=0.8)
    assert out.size == 1 and 1499 <= int(out[0]) <= 1500, out


def test_shoulder_is_stepped_over():
    """A crossing part-way up one event, above a small shoulder dip, still
    walks back to the event's real foot (the 'late' guard)."""
    lp = np.zeros(T)
    lp[1500:1506] = np.linspace(0.0, 1.2, 6)      # rise ...
    lp[1506:1508] = [1.1, 1.05]                   # ... small shoulder dip ...
    lp[1508:1520] = np.linspace(1.1, 2.0, 12)     # ... rise continues
    lp[1520:] = 2.0
    out = refine_onsets(np.array([1510]), lp, FPS, backtrack_s=0.8)
    assert out.size == 1 and 1499 <= int(out[0]) <= 1500, out

if __name__ == "__main__":
    test_backtracks_crossing_to_rise_foot()
    test_absolute_floor_drops_noise_keeps_real()
    test_floor_zeros_out_a_flat_noise_roi()
    test_monotonic_in_floor()
    test_edges_empty_and_dedupe()
    test_foot_not_dragged_onto_flat_baseline()
    test_shoulder_is_stepped_over()
    print("refine_onsets self-check passed")

"""A non-positive baseline must not collapse the dF/F denominator to 1e-9."""
import numpy as np
import pytest

from calliope.core import utils


def _trace(baseline_head):
    """Noisy ~300-count ROI whose corrected signal dips for the first 20 s."""
    rng = np.random.default_rng(0)
    F = 300 + rng.normal(0, 20, 3000).astype(np.float32)
    F[:200] = baseline_head + rng.normal(0, 5, 200)
    return F.astype(np.float32)


def test_first_n_negative_baseline_stays_bounded():
    dff = utils.first_n_min_df_over_f_1d(_trace(-5.0), baseline_min=0.3, fps=10.0)
    assert np.abs(dff).max() < 100


def test_rolling_negative_baseline_stays_bounded():
    F = _trace(300.0)
    F[1000:1600] = -50 + F[1000:1600] * 0.01    # over-subtracted stretch
    dff = utils.robust_df_over_f_1d(F, win_sec=20, perc=10, fps=10.0)
    assert np.abs(dff).max() < 100


@pytest.mark.parametrize("fn", [utils.first_n_min_df_over_f_1d,
                                utils.robust_df_over_f_1d])
def test_healthy_trace_is_unchanged(fn):
    F = _trace(300.0)
    F0 = (np.mean(F[:180]) if fn is utils.first_n_min_df_over_f_1d else None)
    kw = {"baseline_min": 0.3} if F0 is not None else {"win_sec": 20}
    dff = fn(F, fps=10.0, **kw)
    if F0 is not None:
        np.testing.assert_allclose(dff, (F - F0) / F0, rtol=1e-4, atol=1e-5)
    assert np.abs(dff).max() < 1


def test_flat_zero_trace_is_finite():
    dff = utils.first_n_min_df_over_f_1d(np.zeros(500, np.float32), fps=10.0)
    assert np.isfinite(dff).all() and (dff == 0).all()

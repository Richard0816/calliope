"""best_lag_sec sign must match the documented convention: positive => A leads B.

Before 2026-11-09 both routines returned the opposite sign, so every Lead/Lag
violin had its colours swapped.
"""
import numpy as np

from calliope.core.crosscorrelation import batch_xcorr_clusters, single_pair_xcorr_curve

FPS = 10.0


def _a_leads_b(delay_s=2.0, T=3000):
    rng = np.random.default_rng(0)
    a = np.convolve(rng.standard_normal(T + 200), np.ones(15) / 15, "same")[100:-100]
    b = np.roll(a, int(delay_s * FPS))          # B(t) = A(t - delay): A leads
    return a.astype(np.float32), b.astype(np.float32)


def test_batch_positive_when_a_leads():
    a, b = _a_leads_b()
    lag, _, _ = batch_xcorr_clusters(a[:, None], b[:, None], FPS, 5.0, use_gpu=False)
    assert abs(float(lag[0, 0]) - 2.0) < 0.15


def test_single_pair_curve_positive_when_a_leads():
    a, b = _a_leads_b()
    lags, r = single_pair_xcorr_curve(a, b, FPS, 5.0)
    assert np.all(np.diff(lags) > 0)
    assert abs(float(lags[np.argmax(r)]) - 2.0) < 0.15

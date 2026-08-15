"""LOESS behaviour and its measured agreement with R.

Numeric agreement on real DTEval calls is covered by the parity suite; these
pin the guarantees and, importantly, the refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from dteval._loess_direct import loess_direct
from dteval.loess import MultivariateLoessUnavailable, r_loess


def test_univariate_recovers_a_noiseless_signal():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, 200))
    y = 5 + 0.3 * x
    fit = r_loess(x, y)
    assert np.max(np.abs(fit.predict() - y)) < 1e-9


def test_multivariate_interpolate_is_refused_not_guessed():
    """scikit-misc's multivariate fit is wrong; we must not return it."""
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(0, 100, 60), rng.uniform(0, 50, 60)])
    y = x[:, 0] + x[:, 1]
    with pytest.raises(MultivariateLoessUnavailable, match="ehg128|not reproduce R"):
        r_loess(x, y)  # default surface="interpolate"


def test_multivariate_direct_recovers_a_noiseless_plane():
    """The case scikit-misc gets badly wrong: a second predictor that matters."""
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 100, 150)
    v = rng.uniform(0, 50, 150)
    y = 3 + 0.5 * u - 0.25 * v
    got = loess_direct(np.column_stack([u, v]), y, span=0.75, degree=2)
    assert np.max(np.abs(got - y)) < 1e-6, "direct-surface fit should recover a plane"


def test_multivariate_direct_is_reachable_through_r_loess():
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.uniform(0, 10, 80), rng.uniform(0, 10, 80)])
    y = x[:, 0] * 2 + x[:, 1]
    fit = r_loess(x, y, surface="direct")
    values, stderr = fit.predict(se=True)
    assert values.shape == (80,)
    assert np.all(stderr >= 0)


def test_na_rows_are_dropped_and_scattered_back():
    """R's na.omit drops rows; scatter() restores their positions as NaN."""
    x = np.arange(50, dtype="float64")
    y = x * 1.5
    y[7] = np.nan
    fit = r_loess(x, y)
    assert len(fit.used_index) == 49
    out = fit.scatter(fit.predict(), 50)
    assert np.isnan(out[7])
    assert not np.isnan(out[6])

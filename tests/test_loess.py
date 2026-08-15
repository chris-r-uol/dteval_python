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


# ------------------------------------------------------- surfaces vs R ----

import gzip  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "_rcompat.json.gz"


@pytest.fixture(scope="module")
def surfaces():
    if not _FIXTURE.exists():
        pytest.skip("run Rscript parity/generate_rcompat.R")
    with gzip.open(_FIXTURE, "rt") as fh:
        return json.load(fh)["loess_surfaces"]


def test_direct_surface_matches_r_direct(surfaces):
    """Our multivariate LOESS reproduces R's own `surface="direct"`.

    This is the tight check. It isolates the implementation from the separate
    question of how far R's *default* interpolating surface sits from it.
    """
    worst = 0.0
    for block in surfaces:
        jd = np.array([float(v) for v in block["jd"]])
        nn = np.array([float(v) for v in block["nn"]])
        y = np.array([float(v) for v in block["y"]])
        exp = np.array([float(v) for v in block["direct"]])
        got = loess_direct(np.column_stack([jd, nn]), y)
        worst = max(worst, float(np.max(np.abs(got - exp))))
    assert worst < 1e-9, f"direct surface drifted from R's: worst {worst:.3g}"


def test_r_interpolate_is_an_approximation_of_direct(surfaces):
    """Record how far R's default surface sits from the exact local regression.

    `surface="interpolate"` builds a kd-tree and blends between vertices for
    speed; `"direct"` evaluates the fit at every point. The gap below is R's
    approximation error, not ours -- it is why deseason_tube_data cannot match
    R's default output closely, and it is largest for small groups where the
    kd-tree cells are coarse relative to the data.
    """
    gaps = {}
    for block in surfaces:
        n = block["n"] if isinstance(block["n"], int) else block["n"][0]
        direct = np.array([float(v) for v in block["direct"]])
        interp = np.array([float(v) for v in block["interpolate"]])
        gaps[int(n)] = float(np.max(np.abs(direct - interp)))
    assert gaps, "no surface fixtures"
    # Small groups diverge most; this pins the observed behaviour so a change
    # in either direction is visible.
    assert max(gaps.values()) > 0.1, f"expected a visible gap, got {gaps}"
    assert max(gaps.values()) < 20, f"gap larger than ever observed: {gaps}"

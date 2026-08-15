"""The GAM surface: how far the port sits from ``mgcv``, and why.

``fit_tube_model_gam`` is the one model in the port that is close to R rather
than equal to it. The construction is reproduced (cubic-regression-spline
marginals, k = 5, quantile knots on the unique values, tensor product, one
penalty per direction, GCV); mgcv's smoothing-parameter *optimiser* is not.

These tests pin two things so neither can drift unnoticed: the basis is right
(exact properties that hold regardless of the optimiser), and the measured
deviation from R stays within the bound docs/parity.md quotes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from parity.compare import load_fixture, load_value

import dteval as dte
from dteval.gam import _cr_basis, _cr_penalty, _knots, fit_tensor_gam

# The bound docs/parity.md and the manifest quote, with a little headroom.
MAX_ABS_DIFF = 1.5
MAX_MEDIAN_DIFF = 0.05
MIN_CORRELATION = 0.999


@pytest.fixture(scope="module")
def reference():
    try:
        return load_value(load_fixture("fit/gam.location")["value"])
    except FileNotFoundError:
        pytest.skip("run Rscript parity/generate.R fit/gam.location")


def test_knots_use_unique_values():
    """mgcv places cr knots at quantiles of the *unique* covariate values."""
    v = np.array([0.0] * 100 + [1.0, 2.0, 3.0, 4.0])
    assert _knots(v, 5) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_basis_interpolates_at_knots():
    """The cr basis is parameterised by function values, so it is the identity
    at the knots -- the property that makes the penalty cheap to write down."""
    knots = np.linspace(0.0, 1.0, 5)
    assert _cr_basis(knots, knots) == pytest.approx(np.eye(5), abs=1e-12)


def test_penalty_annihilates_straight_lines():
    """The penalty is an integrated squared *second* derivative, so a straight
    line through the knots must cost nothing."""
    knots = np.array([0.0, 0.7, 1.3, 2.9, 4.0])
    penalty = _cr_penalty(knots)
    for beta in (np.ones(5), 3 * knots + 1):
        assert float(beta @ penalty @ beta) == pytest.approx(0.0, abs=1e-9)
    assert float(knots**2 @ penalty @ knots**2) > 1.0


def test_recovers_a_known_smooth_surface():
    """A noiseless tensor-product surface must be recovered to near machine
    precision -- this is about the basis, not the smoothing parameters."""
    rng = np.random.default_rng(20260815)
    x = rng.uniform(-1, 1, size=(600, 2))
    y = 3 + 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 0] * x[:, 1]

    fit = fit_tensor_gam(x, y).predict(x)
    assert np.abs(fit - y).max() < 1e-6


def test_matches_mgcv_within_the_documented_bound(reference):
    """The measured gap against R, which docs/parity.md quotes."""
    got = dte.fit_tube_model(
        dte.datasets.dt_brd(),
        tube=".value",
        inputs=[".longitude", ".latitude"],
        model=dte.fit_tube_model_gam,
    )
    expected = reference[".value.pred"].to_numpy(float)
    actual = got[".value.pred"].to_numpy(float)
    assert len(actual) == len(expected)

    diff = np.abs(actual - expected)
    assert np.median(diff) < MAX_MEDIAN_DIFF
    assert diff.max() < MAX_ABS_DIFF
    assert np.corrcoef(actual, expected)[0, 1] > MIN_CORRELATION


def test_everything_but_the_fit_matches_r_exactly(reference):
    """The gap is confined to the two fitted columns."""
    got = dte.fit_tube_model(
        dte.datasets.dt_brd(),
        tube=".value",
        inputs=[".longitude", ".latitude"],
        model=dte.fit_tube_model_gam,
    )
    assert list(got.columns) == list(reference.columns)

    fitted = {".value.pred", ".value.pred.se"}
    for column in reference.columns:
        if column in fitted:
            continue
        left, right = reference[column], got[column]
        if pd.api.types.is_numeric_dtype(right) and not pd.api.types.is_bool_dtype(right):
            assert np.allclose(
                left.to_numpy("float64", na_value=np.nan),
                right.to_numpy("float64", na_value=np.nan),
                rtol=1e-6,
                equal_nan=True,
            ), column
        else:
            assert list(left.astype(str)) == list(right.astype(str)), column

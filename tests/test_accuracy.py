"""Accuracy behaviour the parity fixtures do not reach.

The fixtures pin ``test_tube_accuracy`` against R for one reference monitor and
a widened ``max.distance``. What they do not exercise is the error paths, the
default 10 m radius (which finds almost nothing, on purpose), and the regression
itself against an answer known independently of R.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import dteval as dte
from dteval.accuracy import _ols
from dteval.handlers import DTEvalError


@pytest.fixture(scope="module")
def tubes():
    return dte.datasets.dt_brd()


@pytest.fixture(scope="module")
def reference():
    return dte.datasets.aurn_example()


def test_ols_matches_scipy_on_a_known_fit():
    """summary(lm(y ~ x)) coefficients and p-values, cross-checked."""
    rng = np.random.default_rng(20260815)
    x = rng.normal(20, 6, 200)
    y = 3.5 + 0.8 * x + rng.normal(0, 2, 200)

    got = _ols(x, y)
    expected = stats.linregress(x, y)

    assert got["slope"] == pytest.approx(expected.slope)
    assert got["intercept"] == pytest.approx(expected.intercept)
    assert got["p_slope"] == pytest.approx(expected.pvalue)
    assert got["adj_r2"] == pytest.approx(1 - (1 - expected.rvalue**2) * 199 / 198)


def test_ols_ignores_missing_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([2.0, 4.0, 99.0, 8.0, 10.0, np.nan])
    fit = _ols(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(0.0, abs=1e-9)


def test_missing_pollutant_is_an_error(tubes, reference):
    with pytest.raises(DTEvalError, match="Expecting 'pm10'"):
        dte.test_tube_accuracy(tubes, reference, ref="pm10", show=[])


def test_no_reference_is_an_error(tubes):
    with pytest.raises(DTEvalError, match="Expecting 'no2'"):
        dte.test_tube_accuracy(tubes, None, show=[])


def test_more_than_one_group_is_an_error(tubes, reference):
    with pytest.raises(DTEvalError, match="only one group term"):
        dte.test_tube_accuracy(
            tubes, reference, max_distance=2500, group=[".year", "site"], show=[]
        )


def test_default_radius_uses_only_the_co_located_tubes(tubes, reference):
    """max.distance defaults to 10 m, so only the tubes sharing the monitor's
    coordinates pair up -- a much smaller fit than the widened radius."""
    tight = dte.test_tube_accuracy(tubes, reference, show=[])
    wide = dte.test_tube_accuracy(tubes, reference, max_distance=2500, show=[])

    assert (tight["data"]["distance.m"] == 0).all()
    assert 0 < len(tight["data"]) < len(wide["data"])
    assert "dist < 10" in tight["report"]


def test_an_unreachable_radius_reports_insufficient_data(tubes, reference):
    """R reports rather than errors when a cut has too few pairs to regress.
    The co-located tubes sit at distance exactly 0, so a radius of 0 excludes
    everything."""
    result = dte.test_tube_accuracy(tubes, reference, max_distance=0, show=[])
    assert "Insufficient data" in result["report"]
    assert result["lookup"]["adj.r.squared"].isna().all()


def test_all_pairs_are_kept_in_all(tubes, reference):
    """`data` is the pairs inside max.distance; `all` is every date-matched
    pair, so it must be the larger of the two."""
    result = dte.test_tube_accuracy(tubes, reference, max_distance=2500, show=[])
    assert len(result["all"]) > len(result["data"])
    assert (result["data"]["distance.m"] < 2500).all()
    assert not result["all"][".ref"].isna().any()


def test_nearest_only_keeps_a_single_distance(tubes, reference):
    result = dte.test_tube_accuracy(
        tubes, reference, max_distance=2500, nearest_only=True, show=[]
    )
    assert result["data"]["distance.m"].nunique() == 1
    assert "nearest.only" in result["report"]


def test_reference_columns_are_suffixed(tubes, reference):
    """R's convention: every reference column arrives as `<name>.ref`."""
    result = dte.test_tube_accuracy(tubes, reference, max_distance=2500, show=[])
    for column in ("no2", "site", "code", "source", "latitude", "longitude"):
        assert f"{column}.ref" in result["data"].columns
    assert result["data"]["source.ref"].eq("synthetic").all()


def test_lookup_matches_a_direct_regression(tubes, reference):
    """The reported coefficients are the OLS fit of the returned pairs."""
    result = dte.test_tube_accuracy(tubes, reference, max_distance=2500, show=[])
    pairs = result["data"]
    direct = _ols(pairs["no2.ref"].to_numpy(float), pairs[".value"].to_numpy(float))

    row = result["lookup"].iloc[0]
    assert row["n"] == len(pairs)
    assert row["slope"] == pytest.approx(direct["slope"])
    assert row["intercept"] == pytest.approx(direct["intercept"])
    assert row["adj.r.squared"] == pytest.approx(direct["adj_r2"])


def test_facets_cut_the_report(tubes, reference):
    result = dte.test_tube_accuracy(
        tubes, reference, max_distance=2500, facet=".year", show=[]
    )
    cuts = result["lookup"][".cut"].tolist()
    assert cuts == sorted(cuts)
    assert len(cuts) == result["data"][".cut"].nunique()


def test_plot_carries_the_lm_smooth(tubes, reference):
    result = dte.test_tube_accuracy(tubes, reference, max_distance=2500, show=[])
    layers = result["plot"].to_spec()["layers"]
    assert [layer["geom"] for layer in layers] == ["GeomPoint", "GeomSmooth"]
    assert layers[1]["params"]["method"] == "lm"
    assert layers[1]["params"]["formula"] == "y~x"
    assert layers[1]["mapping"] == {"x": "no2.ref", "y": ".value"}

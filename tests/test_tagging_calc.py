"""Behavioural tests for tagging, handlers and calc.

Numeric agreement with R is covered by the parity suite. These cover the paths
a fixture cannot capture -- error messages, method dispatch, argument overrides
and the reproduced upstream sharp edges.
"""

from __future__ import annotations

import pandas as pd
import pytest

import dteval as dte
from dteval.handlers import DTEvalError
from dteval.rcompat.reval import RExprError, r_eval


@pytest.fixture
def dt():
    return dte.datasets.dt_brd()


# ------------------------------------------------------------------ tagging


def test_tag_tube_adds_expected_columns_in_r_order(dt):
    out = dte.tag_tube(dt)
    added = [c for c in out.columns if c not in dt.columns]
    assert added == [".latitude", ".longitude", ".sample_id", ".value"]


def test_tag_tube_is_idempotent(dt):
    once = dte.tag_tube(dt)
    twice = dte.tag_tube(once)
    pd.testing.assert_frame_equal(once, twice)


def test_force_rebuilds_an_existing_tag(dt):
    tagged = dte.tag_tube_value(dt)
    tagged[".value"] = -1.0
    unchanged = dte.tag_tube_value(tagged)
    assert (unchanged[".value"] == -1.0).all(), "should not overwrite without force"
    rebuilt = dte.tag_tube_value(tagged, value_force=True)
    assert (rebuilt[".value"] == dt["bias_adjusted_measurement"]).all()


def test_value_source_can_be_redirected(dt):
    out = dte.tag_tube_value(dt, value="latitude")
    assert (out[".value"] == dt["latitude"]).all()


def test_missing_value_source_raises_r_style_error(dt):
    bare = dt.drop(columns=["bias_adjusted_measurement"])
    with pytest.raises(DTEvalError, match=r"\[tagTubeValue\]"):
        dte.tag_tube_value(bare)


def test_missing_latlon_source_raises_r_style_error(dt):
    bare = dt.drop(columns=["latitude"])
    with pytest.raises(DTEvalError, match="expected latlon source"):
        dte.tag_tube_lat_lon(bare)


def test_unknown_method_is_rejected(dt):
    with pytest.raises(DTEvalError, match="unknown method"):
        dte.tag_tube_start_end(dt, method=9, startend_force=True)


def test_start_end_falls_through_to_the_next_method(dt):
    """method=-1 tries each in turn; method 1 needs month + year."""
    bare = dt.drop(columns=[".start_date", ".end_date", "month", "month_numeric"])
    # Only method 3 can work now, and it needs start/end columns -- which are
    # gone, so every method fails and the R-style error surfaces.
    with pytest.raises(DTEvalError, match="failed to match/build"):
        dte.tag_tube_start_end(bare)


def test_date_midpoint_rounds_half_to_even(dt):
    """R's +.Date rounds the difftime, half to even (see docs/parity.md)."""
    frame = pd.DataFrame(
        {
            ".start_date": pd.to_datetime(["2022-03-30", "2022-01-05"]),
            ".end_date": pd.to_datetime(["2022-05-04", "2022-02-07"]),
        }
    )
    got = dte.tag_tube_date(frame)[".date"]
    # 35 days -> 17.5 -> 18 (up to even); 33 days -> 16.5 -> 16 (down to even)
    assert list(got) == [pd.Timestamp("2022-04-17"), pd.Timestamp("2022-01-21")]


def test_month_factor_keeps_all_twelve_levels(dt):
    out = dte.tag_tube_month(dt)
    assert list(out[".month"].cat.categories) == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]


def test_year_is_an_ordered_factor(dt):
    out = dte.tag_tube_year(dt)
    assert out[".year"].cat.ordered


def test_location_uses_as_character_not_format(dt):
    """.location goes through paste (as.character); .sample_id through format."""
    out = dte.tag_tube_location(dt)
    assert out[".location"].iloc[0] == "{53.75975,-1.73278}"


# ----------------------------------------------------------------- handlers


def test_get_tube_x_returns_none_by_default_on_failure(dt):
    assert dte.get_tube_x(dt, "no_such_column") is None


def test_get_tube_x_can_raise_with_the_calling_function_named(dt):
    with pytest.raises(DTEvalError, match=r"\[calcTubeStat\].*tube"):
        dte.get_tube_x(dt, "nope", if_err="stop<<calcTubeStat>>tube")


def test_get_tube_x_enforces_test_class(dt):
    assert dte.get_tube_x(dt, "site", test_class="numeric") is None
    assert dte.get_tube_x(dt, "latitude", test_class="numeric") is not None


def test_check_tube_data_names_columns_after_the_expression(dt):
    out = dte.check_tube_data(dte.tag_tube(dt), ["factor(local_authority)"])
    assert "factor(local_authority)" in out.columns
    assert isinstance(out["factor(local_authority)"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------- expression evaluator


def test_evaluator_rejects_unsupported_functions(dt):
    with pytest.raises(RExprError, match="not in the supported R subset"):
        r_eval("system('rm -rf /')", dt)


def test_evaluator_rejects_python_syntax(dt):
    with pytest.raises(RExprError):
        r_eval("__import__('os').getcwd()", dt)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2", 3.0),
        ("paste('a', 'b')", "a b"),
        ("paste0('a', 'b')", "ab"),
        ("paste('a', 'b', sep='-')", "a-b"),
    ],
)
def test_evaluator_basics(expr, expected):
    got = r_eval(expr, None)
    got = got.iloc[0] if isinstance(got, pd.Series) else got
    assert got == expected


def test_evaluator_handles_backticked_names():
    frame = pd.DataFrame({"Site Type": ["a", "b"]})
    assert list(r_eval("`Site Type`", frame)) == ["a", "b"]


# --------------------------------------------------------------------- calc


def test_comma_in_by_is_rejected_as_in_r(dt):
    """data.table splits a `by` string on commas; R errors, so we must too.

    ``checkTubeData`` happily evaluates ``paste(a, b)`` into a column of that
    literal name, and only then does data.table split it and fail to find the
    pieces. Reproduced deliberately -- see calc.py::_split_by_on_commas.
    """
    with pytest.raises(DTEvalError, match="not found"):
        dte.calc_tube_stat(dt, tube=".value", by="paste(year_of_measurement, month)")


def test_scalar_stat_names_the_column_without_a_suffix(dt):
    out = dte.calc_tube_stat(dt, tube=".value", by="year_of_measurement", stat=lambda x: len(x))
    assert ".value" in out.columns
    assert ".value.mean" not in out.columns


def test_mapping_stat_suffixes_the_column(dt):
    out = dte.calc_tube_stat(dt, tube=".value", by="year_of_measurement")
    assert ".value.mean" in out.columns


def test_no_by_gives_a_single_row(dt):
    out = dte.calc_tube_stat(dt)
    assert len(out) == 1

"""Verify the R-semantics primitives directly against R.

Everything in the port is built on these, so they are checked independently and
first. Reference values come from ``parity/generate_rcompat.R``.

Two kinds of assertion, deliberately:

* **exact** for anything structural -- factor levels and their order, date
  arithmetic, cut labels, counts. Getting these wrong changes the answer, not
  its last digits.
* **to a relative tolerance** for arithmetic. See parity/compare.py for why
  bit-exactness is not the bar.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from dteval.rcompat import collate, dates, factor, stats

FIXTURE = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "_rcompat.json.gz"

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def ref() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; run Rscript parity/generate_rcompat.R")
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


#: Relative tolerance for arithmetic, matching parity/compare.py.
TOL = 1e-6


def g17(v: float) -> str:
    return "%.17g" % float(v)


def close(got: float, exp: float) -> bool:
    """True if the two agree to TOL, treating NaN as equal to NaN."""
    if got != got or exp != exp:
        return got != got and exp != exp
    if got == exp:
        return True
    scale = max(abs(got), abs(exp))
    return abs(got - exp) <= TOL * scale if scale else True


def assert_all_equal(got, exp, label: str, limit: int = 5) -> None:
    got, exp = list(got), list(exp)
    assert len(got) == len(exp), f"{label}: length {len(got)} != {len(exp)}"
    bad = [(i, g, e) for i, (g, e) in enumerate(zip(got, exp, strict=True)) if g != e]
    if bad:
        shown = "\n".join(f"    [{i}] python={g!r} R={e!r}" for i, g, e in bad[:limit])
        extra = f"\n    ... {len(bad) - limit} more" if len(bad) > limit else ""
        raise AssertionError(f"{label}: {len(bad)}/{len(exp)} differ\n{shown}{extra}")


# ---------------------------------------------------------------------- dates


@pytest.mark.parametrize(("key", "fmt"), [("ymd", "%Y-%m-%d"), ("j", "%j"), ("b", "%b"), ("B", "%B")])
def test_fractional_date_formatting(ref, key, fmt):
    """R truncates a fractional Date rather than rounding it."""
    dd = ref["dates"]
    ser = dates.date_from_days([float(v) for v in dd["days"]])
    assert_all_equal(list(dates.r_format_date(ser, fmt)), dd[key], f"format.Date({fmt})")


def test_month_bounds_match_posixlt_trick(ref):
    mb = ref["month_bounds"]
    start = dates.as_numeric_date(dates.first_of_month(mb["year"], mb["month"]))
    end = dates.as_numeric_date(dates.end_of_month(mb["year"], mb["month"]))
    assert_all_equal([g17(v) for v in start], [g17(float(v)) for v in mb["start"]], "start")
    assert_all_equal([g17(v) for v in end], [g17(float(v)) for v in mb["end"]], "end")


# -------------------------------------------------------------- collate/factor


def test_collation_and_factor_levels(ref):
    """Level order determines .sample_id values, so it is a value not a display detail."""
    c = ref["collate"]
    assert_all_equal(collate.r_sort(c["input"]), c["sorted"], "sort()")
    fac = factor.r_factor(c["input"])
    assert_all_equal(list(fac.cat.categories), c["levels"], "factor levels")
    assert_all_equal(
        [g17(v) for v in factor.as_numeric_factor(fac)],
        [g17(float(v)) for v in c["codes"]],
        "as.numeric(factor(.))",
    )


def test_summary_factor_frame(ref):
    sf = ref["summary_factor"]
    got = factor.summary_factor_frame([1] * 4107 + [2] * 410 + [3] * 3111)
    assert_all_equal(list(got.columns), sf["names"], "data.frame(t(summary(factor))) names")
    assert_all_equal([int(v) for v in got.iloc[0]], [int(v) for v in sf["values"]], "counts")


def test_cut_labels_use_percent_g(ref):
    """R builds cut labels with %.3g, so 1000 renders as 1e+03."""
    cu = ref["cut"]
    cc = stats.r_cut(
        [0.5, 5, 50, 500, 5e4, 5e6, 0, None, 1e8],
        [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000],
    )
    assert_all_equal(list(cc.cat.categories), cu["levels"], "cut levels")
    got = [(int(c) + 1 if c >= 0 else None) for c in cc.cat.codes]
    exp = [None if e in (None, "NA") else int(e) for e in cu["codes"]]
    assert_all_equal(got, exp, "cut codes")


# ------------------------------------------------------------------ aggregate


def _agg_mismatches(block, fns) -> dict[str, int]:
    bad = dict.fromkeys(fns, 0)
    for i, xs in enumerate(block["x"]):
        a = np.array([float(v) for v in xs])
        for name, fn in fns.items():
            if not close(fn(a), float(block[name][i])):
                bad[name] += 1
    return bad


def test_aggregation_matches_r_on_real_replicate_groups(ref):
    """mean/sd/var/median agree with R across every real replicate group."""
    block = ref["agg_real"]
    bad = _agg_mismatches(
        block,
        {
            "mean": lambda a: stats.r_mean(a, na_rm=True),
            "sd": lambda a: stats.r_sd(a, na_rm=True),
            "var": lambda a: stats.r_var(a, na_rm=True),
            "median": lambda a: stats.r_median(a, na_rm=True),
        },
    )
    assert bad == dict.fromkeys(bad, 0), f"mismatches over {len(block['x'])} groups: {bad}"


def test_quantile_exact_on_real_replicate_groups(ref):
    block = ref["agg_real"]
    probs = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]
    bad = 0
    for i, xs in enumerate(block["x"]):
        a = np.array([float(v) for v in xs])
        got = stats.r_quantile(a, probs, na_rm=True)
        exp = [float(v) for v in block["q"][i]]
        if not all(close(g, e) for g, e in zip(got, exp, strict=True)):
            bad += 1
    assert bad == 0, f"{bad}/{len(block['x'])} quantile groups differ"


def test_aggregation_matches_r_over_a_wide_numeric_range(ref):
    """Same agreement over an adversarial range: n up to 60, 13 orders of magnitude."""
    block = ref["agg_synthetic"]
    bad = _agg_mismatches(
        block, {"mean": stats.r_mean, "sd": stats.r_sd, "var": stats.r_var}
    )
    assert bad == dict.fromkeys(bad, 0), f"mismatches over {len(block['x'])} samples: {bad}"


# ------------------------------------------------------- distributions ----
def test_qt_matches_r(ref):
    qt = ref["qt"]
    dfs = [float(v) for v in qt["df"]]
    for tail, key in ((False, "upper"), (True, "lower")):
        exp = [float(v) for v in qt[key]]
        for df, e in zip(dfs, exp, strict=True):
            got = float(stats.qt(0.05 / 2, df, lower_tail=tail)[0])
            assert close(got, e), f"qt(df={df}, lower={tail}): python={got!r} R={e!r}"


def test_qnorm_matches_r(ref):
    qn = ref["qnorm"]
    for p, e in zip(qn["p"], [float(v) for v in qn["v"]], strict=True):
        got = float(stats.qnorm(float(p))[0])
        assert close(got, e), f"qnorm({p}): python={got!r} R={e!r}"

"""Verify the R-semantics primitives directly against R.

Everything in the port is built on these, so they are checked independently and
first. Reference values come from ``parity/generate_rcompat.R``.

Bit-exactness is asserted via ``%.17g``, which round-trips every double.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dteval.rcompat import collate, dates, factor, numfmt, rmath, stats

FIXTURE = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "_rcompat.json.gz"

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def ref() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; run Rscript parity/generate_rcompat.R")
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


def g17(v: float) -> str:
    return "%.17g" % float(v)


def assert_all_equal(got, exp, label: str, limit: int = 5) -> None:
    got, exp = list(got), list(exp)
    assert len(got) == len(exp), f"{label}: length {len(got)} != {len(exp)}"
    bad = [(i, g, e) for i, (g, e) in enumerate(zip(got, exp, strict=True)) if g != e]
    if bad:
        shown = "\n".join(f"    [{i}] python={g!r} R={e!r}" for i, g, e in bad[:limit])
        extra = f"\n    ... {len(bad) - limit} more" if len(bad) > limit else ""
        raise AssertionError(f"{label}: {len(bad)}/{len(exp)} differ\n{shown}{extra}")


# --------------------------------------------------------------------- numfmt


def test_as_character_matches_r(ref):
    nf = ref["numfmt"]
    xs = [float.fromhex(b) for b in nf["bits"]]
    assert_all_equal([numfmt.as_character(x) for x in xs], nf["as_character"], "as.character")


@pytest.mark.parametrize("digits", [1, 4, 15])
def test_signif_matches_r(ref, digits):
    nf = ref["numfmt"]
    xs = [float.fromhex(b) for b in nf["bits"]]
    exp = [float.fromhex(b) for b in nf[f"signif{digits}"]]
    assert_all_equal(
        [g17(numfmt.signif(x, digits)) for x in xs],
        [g17(e) for e in exp],
        f"signif(x, {digits})",
    )


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
            got = fn(a)
            exp = float(block[name][i])
            same = g17(got) == g17(exp) or (got != got and exp != exp)
            if not same:
                bad[name] += 1
    return bad


def test_aggregation_exact_on_real_replicate_groups(ref):
    """mean/sd/var/median/quantile must be bit-exact on DTEval's actual domain."""
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
        if [g17(v) for v in got] != [g17(v) for v in exp]:
            bad += 1
    assert bad == 0, f"{bad}/{len(block['x'])} quantile groups differ"


def test_variance_ulp_ambiguity_outside_dteval_domain_is_bounded(ref):
    """Record the known limit rather than pretending it does not exist.

    R's var loop contracts into an FMA, which we reproduce -- exactly, on
    DTEval's domain. For much larger n over a far wider magnitude range the
    compiled loop evidently takes another shape and ~20% of samples differ by
    one ulp. This test pins that down so a regression cannot hide behind it.
    """
    block = ref["agg_synthetic"]
    bad = _agg_mismatches(
        block,
        {"mean": stats.r_mean, "sd": stats.r_sd, "var": stats.r_var},
    )
    n = len(block["x"])
    assert bad["mean"] == 0, f"mean must stay exact everywhere, got {bad['mean']}/{n}"
    for key in ("sd", "var"):
        assert bad[key] <= 0.25 * n, f"{key} drifted beyond the recorded 1-ulp band: {bad[key]}/{n}"
        for i, xs in enumerate(block["x"]):
            a = np.array([float(v) for v in xs])
            got = getattr(stats, f"r_{key}")(a)
            exp = float(block[key][i])
            if g17(got) != g17(exp):
                assert math.isclose(got, exp, rel_tol=1e-15), (
                    f"{key}[{i}] differs by more than an ulp: python={got!r} R={exp!r}"
                )


# ------------------------------------------------------------------- rmath


CLOSED_FORM_DF = (1.0, 2.0, 1e21)


def test_qt_exact_on_closed_form_branches(ref):
    """qt must be bit-exact wherever R uses a closed form.

    That covers everything DTEval actually asks for: it calls
    ``qt(0.025, df = n - 1, lower.tail = FALSE)`` with ``n`` the replicate
    count, and the default ``n = 3`` gives ``df = 2``.
    """
    qt = ref["qt"]
    dfs = [float(v) for v in qt["df"]]
    for tail, key in ((False, "upper"), (True, "lower")):
        exp = [float(v) for v in qt[key]]
        for df, e in zip(dfs, exp, strict=True):
            if df not in CLOSED_FORM_DF:
                continue
            got = rmath.qt_scalar(0.05 / 2, df, lower_tail=tail)
            assert g17(got) == g17(e), f"qt(df={df}, lower={tail}): python={got!r} R={e!r}"


def test_qt_general_branch_within_one_ulp(ref):
    """Outside the closed forms, qt agrees with R to ~1 ulp but not bit-exactly.

    R refines Hill's expansion with a Newton step driven by ``pt``/``dt``.
    Reproducing that bit-for-bit would mean porting R's incomplete beta as
    well; we use scipy there, and its ~1e-16 relative difference lands in the
    last bits of the result. Measured worst case is 5 ulps (at df = 1.5); the
    bound below leaves a little headroom so a genuine regression still trips
    it. Documented in docs/parity.md.
    """
    qt = ref["qt"]
    dfs = [float(v) for v in qt["df"]]
    max_ulps = 0.0
    for tail, key in ((False, "upper"), (True, "lower")):
        exp = [float(v) for v in qt[key]]
        for df, e in zip(dfs, exp, strict=True):
            if df in CLOSED_FORM_DF:
                continue
            got = rmath.qt_scalar(0.05 / 2, df, lower_tail=tail)
            ulps = 0.0 if got == e else abs(got - e) / math.ulp(abs(e))
            max_ulps = max(max_ulps, ulps)
            assert ulps <= 8, (
                f"qt(df={df}, lower={tail}) drifted to {ulps:.1f} ulps: "
                f"python={got!r} R={e!r}"
            )
    assert max_ulps <= 8


def test_qnorm_matches_r(ref):
    qn = ref["qnorm"]
    got = rmath.qnorm(qn["p"])
    assert_all_equal([g17(v) for v in got], [g17(float(v)) for v in qn["v"]], "qnorm")

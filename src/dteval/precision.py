"""Diffusion tube precision -- port of ``R/test.tube.precision.R``.

Precision is estimated by comparing co-located replicate tubes: samples sharing
a location and sampling period. Four interval methods are available, and the
resulting bounds are LOESS-smoothed against the replicate mean to give the
figure and the lookup table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dteval.handlers import DTEvalError, get_tube_x
from dteval.loess import r_loess
from dteval.plots.ggshell import add_geom, tidy_args
from dteval.plots.tube_plot import TubePlot, tube_plot
from dteval.rcompat.collate import r_sort, r_unique
from dteval.rcompat.numfmt import as_character, signif
from dteval.rcompat.stats import is_finite, qnorm, qt, r_mean, r_quantile, r_sd
from dteval.tagging import tag_tube, tag_tube_required

__all__ = ["test_tube_precision"]

#: Interval methods, as documented in ?testTubePrecision.
METHODS = (1, 2, 3, 4)


def test_tube_precision(
    data: pd.DataFrame,
    tube: str = ".value",
    n: int = 3,
    method: int = 2,
    show: tuple[str, ...] | list[str] = ("plot", "summary.report"),
    **kwargs,
) -> dict[str, Any]:
    """Port of ``testTubePrecision``.

    Methods (``test.tube.precision.R:31``):

    1. ``quantile(x, c(0.025, 0.975))``
    2. ``mean +/- qt(0.025, n-1, lower.tail = FALSE) * sd/sqrt(n)`` (default)
    3. ``mean +/- 1.96 * sd/sqrt(n)``
    4. ``mean +/- qnorm(0.975) * sd``

    Returns ``{"data", "plot", "lookup", "report"}``, matching R's list.
    """
    xargs = {"auto.text": True, "xlab": "replicate mean", "ylab": tube, **kwargs}

    data = tag_tube(data, **kwargs)
    required = [tube] + [v for v in xargs.values() if isinstance(v, str)]
    data = tag_tube_required(data, required=required, **kwargs)

    data = data.copy()
    data[".tube"] = get_tube_x(
        data, tube, test_class="numeric", if_err="stop<<testTubePrecision>>tube"
    )

    group = xargs.get("group")
    group = [] if group is None else ([group] if isinstance(group, str) else list(group))
    if len(group) > 1:
        raise DTEvalError("[testTubePrecision]> Sorry, only one group term allowed \n")

    facet = xargs.get("facet")
    facet = [] if facet is None else ([facet] if isinstance(facet, str) else list(facet))
    if len(facet) > 2:
        raise DTEvalError("[testTubePrecision]> Sorry, no more than two facet terms allowed \n")

    for term in group:
        data[term] = get_tube_x(data, term, if_err="stop<<testTubePrecision>>group")
    for term in facet:
        data[term] = get_tube_x(data, term, if_err="stop<<testTubePrecision>>facet")

    data[".cut"] = _build_cut(data, group, facet)

    if int(method) not in METHODS:
        raise DTEvalError(
            f"[testTubePrecision]> '{method}' unknown method"
            f"\n\trecommend one of: {', '.join(str(m) for m in METHODS)}"
            "\n\t(and maybe check ?testTubePrecision) \n"
        )

    by_cols = [".start_date", ".end_date", ".sample_id", ".cut", *group, *facet]

    chunks, lookups, reports = [], [], []
    # R sorts the cut levels, which fixes the order of the output rows.
    for cut in r_sort(r_unique(data[".cut"])):
        part = data[data[".cut"] == cut]
        chunk, lookup, report = _one_cut(part, cut, by_cols, int(method), n)
        if chunk is not None:
            chunks.append(chunk)
        lookups.append(lookup)
        reports.append(report)

    test = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=[*data.columns, ".n", ".mean"])
    )
    lookup = pd.concat(lookups, ignore_index=True)
    report = "\n".join(reports)
    report = f"'{tube}' (rep = {n} subset):\n{report}"

    plot = _build_plot(test, xargs, n) if len(test) > n else None

    show = [s.lower() for s in show]
    if "summary.report" in show:
        lines = [ln for ln in report.split("\n") if "Insufficient replicates" not in ln]
        print("\n".join(lines) if len(lines) > 1 else "not enough replicated data...")
    if "report" in show:
        print(report)

    return {"data": test, "plot": plot, "lookup": lookup, "report": report}


def _build_cut(data: pd.DataFrame, group: list[str], facet: list[str]) -> pd.Series:
    """R's ``.cut`` key: the subset label each test is computed within."""
    if not group and not facet:
        return pd.Series(["|all|"] * len(data), index=data.index)

    from dteval.rcompat.coerce import as_character_series

    out = pd.Series([""] * len(data), index=data.index)
    for term in ([group[0]] if group else []) + facet[:2]:
        out = out + "|" + as_character_series(data[term]).fillna("NA")
    return out + "|"


def _one_cut(part: pd.DataFrame, cut: str, by_cols: list[str], method: int, n: int):
    """Compute replicate statistics and LOESS bounds for one ``.cut`` subset."""
    stats = _replicate_stats(part, by_cols, method)

    merged = _r_merge(part, stats, by_cols)

    merged = merged[merged[".n"] == n]
    merged = merged[~merged[".tube"].isna()].reset_index(drop=True)

    if len(merged) < n:
        lookup = pd.DataFrame(
            {
                ".cut": [cut],
                "ans": [np.nan],
                "low": [np.nan],
                "high": [np.nan],
                "low.pc": [np.nan],
                "high.pc": [np.nan],
            }
        )
        return None, lookup, f"  {cut} Insufficient replicates..."

    mean = merged[".mean"].to_numpy(dtype="float64")
    fit = r_loess(mean, merged[".tube"].to_numpy(dtype="float64"))
    fit_low = r_loess(mean, merged[".low"].to_numpy(dtype="float64"))
    fit_high = r_loess(mean, merged[".high"].to_numpy(dtype="float64"))

    merged[".y"] = fit.scatter(fit.predict(), len(merged))
    merged[".ylow"] = fit_low.scatter(fit_low.predict(), len(merged))
    merged[".yhigh"] = fit_high.scatter(fit_high.predict(), len(merged))

    # Lookup at each round 10 within the observed mean range.
    grid = np.arange(1, 101, dtype="float64") * 10
    grid = grid[(grid >= np.nanmin(mean)) & (grid <= np.nanmax(mean))]
    lookup = pd.DataFrame({".cut": [cut] * len(grid), "ans": grid})
    lookup["low"] = fit_low.predict(grid)
    lookup["high"] = fit_high.predict(grid)
    lookup["low.pc"] = ((lookup["low"] - lookup["ans"]) / lookup["ans"]) * 100
    lookup["high.pc"] = ((lookup["high"] - lookup["ans"]) / lookup["ans"]) * 100

    low = merged[".low"].to_numpy(dtype="float64")
    high = merged[".high"].to_numpy(dtype="float64")
    report = (
        f"  {cut} mean: {_sig(r_mean(mean, na_rm=True))}"
        f" ({_sig(np.nanmin(mean))} to {_sig(np.nanmax(mean))})"
        f" precision: {_sig(r_mean(((low - mean) / mean) * 100, na_rm=True))}[%]"
        f" to {_sig(r_mean(((high - mean) / mean) * 100, na_rm=True))}[%]"
    )
    return merged, lookup, report


def _r_merge(x: pd.DataFrame, y: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Base R's ``merge(x, y)`` -- column order and, crucially, row order.

    Two behaviours that pandas does not share:

    * the join columns come **first** in the result, then x's remaining
      columns, then y's; and
    * the result is sorted on the join key **as text**. ``merge.data.frame``
      pastes the by-columns into one ``"\\r"``-separated string and orders on
      that, so a numeric key sorts lexicographically: ``.sample_id`` runs
      1, 1151, 1199, ... and 48 lands after 1425, not second.

    Getting this wrong reorders every row of ``testTubePrecision``'s output.

    Note also that R is called without ``by=``, so the join columns are
    ``intersect(names(x), names(y))`` -- ordered by where they sit in **x**,
    not by the order the caller happened to build them in. With
    ``facet = ".year"`` that puts ``.year`` ahead of ``.cut``.
    """
    by = [c for c in x.columns if c in set(by) & set(y.columns)]

    out = x.merge(y, on=by, how="inner")
    rest_x = [c for c in x.columns if c not in by]
    rest_y = [c for c in y.columns if c not in by]
    out = out[[*by, *rest_x, *rest_y]]

    from dteval.rcompat.coerce import as_character_series

    key = None
    for col in by:
        part = as_character_series(out[col]).fillna("NA")
        key = part if key is None else key + "\r" + part
    order = np.argsort(np.asarray(key, dtype=object), kind="stable")
    return out.iloc[order].reset_index(drop=True)


def _sig(value: float) -> str:
    """``signif(x, 4)`` rendered as R's ``paste()`` would render it."""
    return as_character(signif(float(value), 4))


def _replicate_stats(part: pd.DataFrame, by_cols: list[str], method: int) -> pd.DataFrame:
    """Per-replicate-set count, mean and interval, by method."""
    records = []
    grouped = part.groupby(by_cols, sort=False, dropna=False, observed=True)
    for key, chunk in grouped:
        values = chunk[".tube"].to_numpy(dtype="float64")
        finite = values[is_finite(values)]
        row = dict(zip(by_cols, key if isinstance(key, tuple) else (key,), strict=True))
        row[".n"] = len(finite)
        row[".mean"] = r_mean(values, na_rm=True)

        if method == 1:
            q = r_quantile(values, [0.025, 0.975], na_rm=True)
            row[".low"], row[".high"] = q[0], q[1]
        else:
            row[".sd"] = r_sd(values, na_rm=True)
        records.append(row)

    stats = pd.DataFrame.from_records(records)
    if stats.empty:
        return stats

    # Group keys come back as plain objects, which would demote a factor to
    # strings on merge -- and factor level order is part of the output.
    from dteval.calc import _restore_dtype

    for col in by_cols:
        stats[col] = _restore_dtype(stats[col], part[col])

    if method != 1:
        count = stats[".n"].to_numpy(dtype="float64")
        mean = stats[".mean"].to_numpy(dtype="float64")
        sd = stats[".sd"].to_numpy(dtype="float64")
        # R computes the half-width for every row and only then blanks out the
        # n <= 2 cases with ifelse(), so a qt() warning at df = 0 is expected
        # (and suppressed) rather than avoided.
        with np.errstate(invalid="ignore", divide="ignore"):
            if method == 2:
                half = qt(0.05 / 2, count - 1, lower_tail=False) * (sd / np.sqrt(count))
            elif method == 3:
                half = 1.96 * sd / np.sqrt(count)
            else:
                half = qnorm(0.975) * sd
        keep = count > 2
        stats[".low"] = np.where(keep, mean - half, np.nan)
        stats[".high"] = np.where(keep, mean + half, np.nan)

    return stats


def _build_plot(test: pd.DataFrame, xargs: dict, n: int) -> TubePlot:
    """The precision figure: points plus the LOESS fit and its bounds.

    R draws the central fit as a solid red path and the two bounds as dashed
    red paths (``test.tube.precision.R:405``), over whatever ``tubePlot``
    produced.
    """
    base_args = {k: v for k, v in xargs.items() if not k.startswith("line")}
    plot = tube_plot(test, x=".mean", y=".tube", **base_args)

    line_args = tidy_args({k: v for k, v in xargs.items() if k != "colour"}, "line")
    ordered = test.sort_values(".mean", kind="stable")

    for column, linetype in ((".y", "solid"), (".ylow", "dashed"), (".yhigh", "dashed")):
        plot.layers.append(
            add_geom(
                {**line_args, "x": ".mean", "y": column},
                ordered,
                "GeomPath",
                defaults={"colour": "red", "linetype": linetype},
            )
        )
    return plot

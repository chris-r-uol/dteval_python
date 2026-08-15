"""Diffusion tube accuracy -- port of ``R/test.tube.accuracy.R``.

Accuracy is assessed by comparing tubes against a nearby reference monitor
(continuous analyser). The reference series is averaged over each tube's own
sampling window, paired by location, and regressed against the tube values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _sps

from dteval.handlers import DTEvalError, get_tube_x
from dteval.plots.ggshell import add_geom, tidy_args
from dteval.plots.tube_plot import TubePlot, tube_plot
from dteval.rcompat.coerce import as_character_series
from dteval.rcompat.collate import r_sort, r_unique
from dteval.rcompat.numfmt import as_character, signif
from dteval.tagging import tag_tube, tag_tube_required

__all__ = ["test_tube_accuracy"]


def test_tube_accuracy(
    data: pd.DataFrame,
    data_ref: pd.DataFrame | None = None,
    tube: str = ".value",
    ref: str = "no2",
    method: int = 1,
    max_distance: float = 10,
    nearest_only: bool = False,
    show: tuple[str, ...] | list[str] = ("plot", "summary.report"),
    **kwargs,
) -> dict[str, Any]:
    """Port of ``testTubeAccuracy``.

    ``data_ref`` is an openair-shaped reference data set -- a ``date`` column
    plus the pollutant, ``latitude`` and ``longitude``. Its columns arrive in
    the result suffixed ``.ref``, which is R's convention, so a facet or group
    on a reference column has to name the suffixed form.

    Returns ``{"data", "all", "plot", "lookup", "report"}``: ``data`` is the
    pairs closer than ``max_distance``, ``all`` every date-matched pair.
    """
    if data_ref is None or ref not in data_ref.columns:
        raise DTEvalError(f"[testTubeAccuracy]> Expecting '{ref}' in supplied data.ref \n")

    xargs = {"auto.text": True, **kwargs}
    data = tag_tube(data, **kwargs)
    required = [tube] + [v for v in xargs.values() if isinstance(v, str)]
    data = tag_tube_required(data, required=required, **kwargs)

    data = data.copy()
    data[".tube"] = get_tube_x(
        data, tube, test_class="numeric", if_err="stop<<testTubeAccuracy>>tube"
    )

    frames = []
    site_keys = r_unique(
        as_character_series(data_ref["latitude"]).fillna("NA")
        + "<>"
        + as_character_series(data_ref["longitude"]).fillna("NA")
    )
    for key in site_keys:
        frames.append(_one_reference_site(data, data_ref, key, tube, ref, method, xargs))

    out = pd.concat(frames, ignore_index=True)
    ref_col = f"{ref}.ref"

    # Unpaired rows are dropped: they carry no information and only produce a
    # missing-values warning from the plot.
    out = out[~out[".tube"].isna() & ~out[".ref"].isna()].reset_index(drop=True)
    if len(out) < 1:
        raise DTEvalError(
            "[testTubeAccuracy]> Halting test; no tube/data.ref pairs (check sources?)"
        )

    group = xargs.get("group")
    group = [] if group is None else ([group] if isinstance(group, str) else list(group))
    if len(group) > 1:
        raise DTEvalError("[testTubeAccuracy]> Sorry, only one group term allowed \n")
    facet = xargs.get("facet")
    facet = [] if facet is None else ([facet] if isinstance(facet, str) else list(facet))
    if len(facet) > 2:
        raise DTEvalError("[testTubeAccuracy]> Sorry, no more than two facet terms allowed \n")

    for term in group:
        out[term] = get_tube_x(out, term, if_err="stop<<testTubeAccuracy>>group")
    for term in facet:
        out[term] = get_tube_x(out, term, if_err="stop<<testTubeAccuracy>>facet")

    out[".cut"] = _build_cut(out, group, facet)

    tests, locals_, lookups, reports, fitted = [], [], [], [], []
    for cut in r_sort(r_unique(out[".cut"])):
        chunk = out[out[".cut"] == cut]
        near = chunk[chunk["distance.m"] < max_distance]
        if len(near):
            near = near[~near[".tube"].isna() & ~near[".ref"].isna()]
            if nearest_only and len(near):
                near = near[near["distance.m"] == near["distance.m"].min()]

        if len(near) > 3:
            fit = _ols(near[ref_col].to_numpy(float), near[tube].to_numpy(float))
            lookup = pd.DataFrame(
                {
                    ".cut": [cut],
                    "n": pd.array([len(near)], dtype="Int32"),
                    "max.distance": [float(max_distance)],
                    "adj.r.squared": [fit["adj_r2"]],
                    "intercept": [fit["intercept"]],
                    "slope": [fit["slope"]],
                    "p.intercept": [fit["p_intercept"]],
                    "p.slope": [fit["p_slope"]],
                }
            )
            report = (
                f"  {cut} {_sig(fit['intercept'])}\t+ {_sig(fit['slope'])}*[ref]"
                f"\t(adj.r^2 {_sig(fit['adj_r2'])})"
            )
        else:
            lookup = pd.DataFrame(
                {
                    ".cut": [cut],
                    "max.distance": [float(max_distance)],
                    "n": pd.array([len(near)], dtype="Int32"),
                    "adj.r.squared": [np.nan],
                    "intercept": [np.nan],
                    "slope": [np.nan],
                    "p.intercept": [np.nan],
                    "p.slope": [np.nan],
                }
            )
            report = f"  {cut} Insufficient data..."

        tests.append(chunk)
        locals_.append(near)
        lookups.append(lookup)
        reports.append(report)
        fitted.append(len(near) > 3)

    test = pd.concat(tests, ignore_index=True)
    local = pd.concat(locals_, ignore_index=True)
    lookup = pd.concat(lookups, ignore_index=True)
    if not any(fitted):
        # R writes a bare `NA` -- which is *logical* -- when no cut had enough
        # data, and only rbind with a fitted row promotes it to numeric.
        for column in ("adj.r.squared", "intercept", "slope", "p.intercept", "p.slope"):
            lookup[column] = pd.array([None] * len(lookup), dtype="boolean")
    report = "\n".join(reports)
    suffix = "; nearest.only" if nearest_only else ""
    report = f"'{tube}' vs. '{ref_col}' (dist < {as_character(max_distance)}{suffix}):\n{report}"

    plot = _build_plot(local, xargs, ref_col, tube) if len(local) >= 3 else None

    show = [s.lower() for s in show]
    if "summary.report" in show:
        lines = [ln for ln in report.split("\n") if "Insufficient data" not in ln]
        print("\n".join(lines) if len(lines) > 1 else "nothing near enough...")
    if "report" in show:
        print(report)

    return {"data": local, "all": test, "plot": plot, "lookup": lookup, "report": report}


def _one_reference_site(data, data_ref, key, tube, ref, method, xargs) -> pd.DataFrame:
    """Distance-tag every tube against one monitor, then join the aggregated series."""
    try:
        from aqeval import calc_date_range_stat, find_near_lat_lon
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "test_tube_accuracy needs the aqeval package. Install it with: "
            "pip install 'dteval[aqeval]'"
        ) from exc

    ref_key = (
        as_character_series(data_ref["latitude"]).fillna("NA")
        + "<>"
        + as_character_series(data_ref["longitude"]).fillna("NA")
    )
    site = data_ref[ref_key == key]
    first = site.iloc[0]

    # Distance from this monitor to every tube, kept in the tubes' own order.
    tube_ll = data[[".latitude", ".longitude"]].rename(
        columns={".latitude": "latitude", ".longitude": "longitude"}
    )
    tube_ll = tube_ll.assign(cheat=np.arange(1, len(tube_ll) + 1))
    near = find_near_lat_lon(
        float(first["latitude"]), float(first["longitude"]), ref=tube_ll, nmax=len(tube_ll)
    )
    distances = near.sort_values("cheat")["distance_m"].to_numpy()

    out = data.copy()
    out["distance.m"] = distances

    # Average the reference series over each distinct sampling window.
    windows = out[~out[[".start_date", ".end_date"]].duplicated()]
    aggregated = calc_date_range_stat(
        site,
        from_date=list(windows[".start_date"]),
        to_date=list(windows[".end_date"]),
        stat=xargs.get("stat"),
    )
    aggregated = aggregated.rename(
        columns={aggregated.columns[0]: ".start_date", aggregated.columns[1]: ".end_date"}
    )
    for column in ("source", "site", "code"):
        if column in site.columns:
            aggregated[column] = first[column]

    # R suffixes every reference column except the two join keys, so the output
    # always names them consistently rather than only on a clash.
    aggregated = aggregated.rename(
        columns={
            c: f"{c}.ref" for c in aggregated.columns if c not in (".start_date", ".end_date")
        }
    )
    aggregated[".start_date"] = pd.to_datetime(aggregated[".start_date"]).dt.normalize()
    aggregated[".end_date"] = pd.to_datetime(aggregated[".end_date"]).dt.normalize()

    merged = out.merge(aggregated, how="left", on=[".start_date", ".end_date"], sort=False)
    # data.table::merge puts the join columns first; pandas leaves them in place.
    keys = [".start_date", ".end_date"]
    merged = merged[keys + [c for c in merged.columns if c not in keys]]
    merged[".tube"] = merged[tube]
    merged[".ref"] = merged[f"{ref}.ref"]
    return merged


def _build_cut(data: pd.DataFrame, group: list[str], facet: list[str]) -> pd.Series:
    if not group and not facet:
        return pd.Series(["|all|"] * len(data), index=data.index)
    out = pd.Series([""] * len(data), index=data.index)
    for term in ([group[0]] if group else []) + facet[:2]:
        out = out + "|" + as_character_series(data[term]).fillna("NA")
    return out + "|"


def _ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """``summary(lm(y ~ x))`` -- the coefficients and their p-values.

    Written out rather than pulled from statsmodels to keep the dependency
    list to numpy/pandas/scipy.
    """
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    design = np.column_stack([np.ones(n), x])

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    df = n - 2
    sigma2 = float(resid @ resid) / df

    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t = beta / se
    p = 2 * _sps.t.sf(np.abs(t), df)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / df

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "p_intercept": float(p[0]),
        "p_slope": float(p[1]),
        "adj_r2": float(adj_r2),
    }


def _sig(value: float) -> str:
    return as_character(signif(float(value), 4))


def _build_plot(local: pd.DataFrame, xargs: dict, ref_col: str, tube: str) -> TubePlot:
    """Scatter of tube against reference, with an ``lm`` fit over it."""
    base = tidy_args(dict(xargs))
    base = {k: v for k, v in base.items() if not k.startswith("smooth") and k != "..test"}
    plot = tube_plot(local, x=ref_col, y=tube, **base)

    smooth = tidy_args(dict(xargs), "smooth")
    if "group" not in smooth:
        smooth.pop("colour", None)
        smooth.pop("fill", None)
    plot.layers.append(
        add_geom(
            {**smooth, "x": ref_col, "y": tube},
            local,
            "GeomSmooth",
            defaults={"method": "lm", "formula": "y~x", "fill": "grey"},
        )
    )
    return plot

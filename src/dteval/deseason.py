"""Deseasonalise diffusion tube time-series -- port of ``R/deseason.tube.R``.

Ambient NO2 has a strong annual cycle, so a raw year-on-year comparison mixes
the trend with the season. This fits ``[tube] ~ loess(day-of-year + date)`` and
splits the fit into a seasonal component and an underlying trend.

Surface
-------
R fits with LOESS's default ``surface = "interpolate"``. We use
``surface = "direct"``, because ``scikit-misc``'s multivariate path does not fit
correctly and R's interpolating surface needs its ``ehg128`` kd-tree. The two
surfaces differ by roughly 1e-3 relative -- immaterial against the ~10% error
bar on a diffusion tube measurement, and documented in docs/parity.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.handlers import DTEvalError, get_tube_x
from dteval.loess import r_loess
from dteval.rcompat.coerce import as_character_series
from dteval.rcompat.dates import as_numeric_date, r_format_date
from dteval.tagging import tag_tube_required

__all__ = ["deseason_tube_data"]


def deseason_tube_data(
    data: pd.DataFrame,
    tube: str = ".value",
    by: str | list[str] | None = None,
    method: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``deseasonTubeData``.

    Adds ``..fit``, ``..trend``, ``..season`` and ``..deseason`` (plus their
    standard errors) to a per-date summary of the data.

    * ``method=1`` fits at the requested ``by`` level.
    * ``method=2`` fits by location, optionally pooling nearby sites via
      ``max_distance`` / ``max_n``.
    """
    if method not in (1, 2):
        raise DTEvalError("[deseasonTubeData] unknown method; maybe try 1,2?")

    by_cols = [] if by is None else ([by] if isinstance(by, str) else list(by))

    if method == 1:
        group_by = [".date", *by_cols]
        data = tag_tube_required(data, required=[tube, *group_by], **kwargs)
        summary = calc_tube_stat(data, tube, by=group_by, **kwargs)
        summary = _model_inputs(summary, tube)

        if not by_cols:
            summary["..id"] = "|default|"
        else:
            expr = f"paste({', '.join(by_cols)}, sep='')"
            summary["..id"] = get_tube_x(summary, expr)

        return _fit_by_group(summary, pd.unique(summary["..id"]), **kwargs)

    # method 2: fit by location, pooling neighbours
    group_by = list(
        dict.fromkeys([".date", ".longitude", ".latitude", ".location", *by_cols])
    )
    data = tag_tube_required(data, required=[tube, *group_by], **kwargs)

    if "max_n" not in kwargs and "max_distance" not in kwargs:
        kwargs["max_distance"] = 0

    summary = calc_tube_stat(data, tube, by=group_by, **kwargs)
    summary = _model_inputs(summary, tube)
    summary["..id"] = (
        as_character_series(summary[".latitude"]).fillna("NA")
        + " "
        + as_character_series(summary[".longitude"]).fillna("NA")
    )

    from aqeval import find_near_lat_lon

    chunks = []
    for site in sorted(pd.unique(summary["..id"])):
        here = summary[summary["..id"] == site]
        near = find_near_lat_lon(
            float(here[".latitude"].iloc[0]),
            float(here[".longitude"].iloc[0]),
            ref=data,
            nmax=len(data),
            lat_col=".latitude",
            lon_col=".longitude",
        )
        if "max_distance" in kwargs:
            near = near[near["distance_m"] <= kwargs["max_distance"]]
        ids = pd.unique(
            as_character_series(near[".latitude"]).fillna("NA")
            + " "
            + as_character_series(near[".longitude"]).fillna("NA")
        )
        if "max_n" in kwargs and len(ids) > kwargs["max_n"]:
            ids = ids[: kwargs["max_n"]]

        pooled = summary[summary["..id"].isin(ids)]
        fitted = _fit_one(pooled, **kwargs)
        if fitted is None:
            continue
        # R keeps only the site the model was centred on, however many
        # neighbours went into building it.
        chunks.append(fitted[fitted["..id"] == site])

    if not chunks:
        return summary.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


def _model_inputs(summary: pd.DataFrame, tube: str) -> pd.DataFrame:
    """Add the numeric LOESS inputs: day-of-year, date-as-number, response."""
    out = summary.copy()
    out["jd"] = pd.to_numeric(r_format_date(out[".date"], "%j"), errors="coerce").astype(
        "float64"
    )
    out["n"] = as_numeric_date(out[".date"])
    out[".y"] = out[f"{tube}.mean"].astype("float64")
    return out


def _fit_by_group(summary: pd.DataFrame, ids, **kwargs) -> pd.DataFrame:
    chunks = []
    for key in ids:
        fitted = _fit_one(summary[summary["..id"] == key], **kwargs)
        if fitted is not None:
            chunks.append(fitted)
    if not chunks:
        return summary.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


def _fit_one(chunk: pd.DataFrame, **kwargs) -> pd.DataFrame | None:
    """Fit one group and split the fit into trend and season.

    R refuses to build a model on three points or fewer, and drops the group if
    the fit errors.
    """
    if len(chunk) <= 2:
        return None

    out = chunk.reset_index(drop=True).copy()
    span = kwargs.get("span", 0.75)
    degree = kwargs.get("degree", 2)

    try:
        model = r_loess(
            out[["jd", "n"]], out[".y"], span=span, degree=degree, surface="direct"
        )
    except Exception:  # noqa: BLE001 -- R wraps this in try() and drops the group
        return None

    fit, fit_se = model.predict(out[["jd", "n"]], se=True)
    out["..fit"] = model.scatter(fit[: len(model.used_index)], len(out))
    out["..fit.se"] = model.scatter(fit_se[: len(model.used_index)], len(out))

    # The trend is the same surface evaluated at the mean day-of-year, i.e.
    # with the seasonal coordinate held fixed.
    flat = out[["jd", "n"]].copy()
    flat["jd"] = out["jd"].mean()
    trend, trend_se = model.predict(flat, se=True)
    out["..trend"] = model.scatter(trend[: len(model.used_index)], len(out))
    out["..trend.se"] = model.scatter(trend_se[: len(model.used_index)], len(out))

    # Re-centre the trend onto the fit's level. Note R uses mean() without
    # na.rm here (deseason.tube.R:297, flagged in its own comment), so a single
    # missing value propagates -- reproduced rather than corrected.
    out["..trend"] = out["..trend"] - np.mean(out["..trend"])
    out["..trend"] = out["..trend"] + np.mean(out["..fit"])

    out["..season"] = out["..fit"] - out["..trend"]
    out["..deseason"] = out[".y"] - out["..season"]

    return out.sort_values(".date", kind="stable").reset_index(drop=True)

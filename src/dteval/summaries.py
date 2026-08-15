"""Diffusion tube data summaries -- port of ``R/tube.summaries.R``.

Overview statistics for a tagged DT data set: how much data, over what period,
from how many locations, and how much of it looks suspect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dteval.rcompat.coerce import as_character_series
from dteval.rcompat.factor import r_summary_factor, summary_factor_frame
from dteval.rcompat.stats import r_cut, r_median
from dteval.tagging import tag_tube

__all__ = ["tube_summary", "tube_summary_lat_lon", "tube_summary_sample"]

#: Distance bands used by tubeSummaryLatLon's brief report.
_LATLON_BREAKS = (0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000)


def tube_summary(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Port of ``tubeSummary`` -- a one-row overview of the data set.

    ``n_bad_tags`` counts samples with a missing coordinate or date, via
    :func:`tube_summary_sample`'s checksum.
    """
    d2 = tag_tube(data, **kwargs)

    n_total = len(d2)
    sampling_from = d2[".start_date"].min()
    sampling_to = d2[".end_date"].max()

    # R counts distinct pasted strings, not tuples -- same answer here, but
    # keep the construction identical in case of odd formatting.
    intervals = as_character_series(d2[".start_date"]) + " " + as_character_series(d2[".end_date"])
    n_intervals = intervals.nunique(dropna=False)

    locations = as_character_series(d2[".latitude"]) + " " + as_character_series(d2[".longitude"])
    n_locations = locations.nunique(dropna=False)

    n_samples = d2[".sample_id"].nunique(dropna=False)

    full = tube_summary_sample(d2, output="full.report", **kwargs)
    n_bad_tags = int((full["checksum"] > 0).sum())

    return pd.DataFrame(
        {
            "sampling.from": [sampling_from],
            "sampling.to": [sampling_to],
            "n.total": pd.array([n_total], dtype="Int32"),
            "n.samples": pd.array([n_samples], dtype="Int32"),
            "n.intervals": pd.array([n_intervals], dtype="Int32"),
            "n.locations": pd.array([n_locations], dtype="Int32"),
            "n.bad.tags": pd.array([n_bad_tags], dtype="Int32"),
        }
    )


def tube_summary_sample(
    data: pd.DataFrame, output: str | None = None, **kwargs
) -> pd.DataFrame:
    """Port of ``tubeSummarySample`` -- per-sample counts and a missing-data checksum.

    ``output='report'`` / ``'full.report'`` returns the per-sample table.
    Otherwise it returns R's brief form: one row of counts of replicate-set
    sizes, with the ``X1``/``X2``/``X3`` column names ``data.frame`` gives a
    transposed ``summary(factor(...))``.
    """
    d2 = tag_tube(data, **kwargs)

    records = []
    # data.table's `by=` returns groups in order of first appearance, not
    # sorted, and this table is not pre-sorted.
    for sample_id, chunk in d2.groupby(".sample_id", sort=False, dropna=False):
        records.append(
            {
                ".sample_id": sample_id,
                "n": len(chunk),
                "missing.latitudes": int(chunk[".latitude"].isna().sum()),
                "missing.longitudes": int(chunk[".longitude"].isna().sum()),
                "missing.start.dates": int(chunk[".start_date"].isna().sum()),
                "missing.end.dates": int(chunk[".end_date"].isna().sum()),
            }
        )
    out = pd.DataFrame.from_records(records)
    for col in ("n", "missing.latitudes", "missing.longitudes",
                "missing.start.dates", "missing.end.dates"):
        out[col] = out[col].astype("Int32")
    out["checksum"] = (
        out["missing.latitudes"]
        + out["missing.longitudes"]
        + out["missing.start.dates"]
        + out["missing.end.dates"]
    )

    if output is not None and str(output).lower() in ("report", "full.report"):
        return out
    return summary_factor_frame(out["n"].astype("int64"))


def tube_summary_lat_lon(
    data: pd.DataFrame, output: str | None = None, **kwargs
) -> Any:
    """Port of ``tubeSummaryLatLon`` -- sample locations ranked by distance from centre.

    The "centre" is the median latitude/longitude of the distinct sampling
    locations; distances come from ``AQEval::findNearLatLon`` (here the
    ``aqeval`` package), and the result is ordered furthest-first so outliers --
    typically mis-keyed coordinates -- surface at the top.

    ``output='report'`` / ``'full.report'`` returns the ranked table; otherwise
    a banded count, as R's ``summary(cut(...))``.
    """
    try:
        from aqeval import find_near_lat_lon
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "tube_summary_lat_lon needs the aqeval package. Install it with: "
            "pip install 'dteval[aqeval]'"
        ) from exc

    d2 = tag_tube(data, **kwargs)

    key = as_character_series(d2[".latitude"]) + " " + as_character_series(d2[".longitude"])
    d2 = d2[~key.duplicated()]
    d2 = d2[~d2[".latitude"].isna()]
    d2 = d2[~d2[".longitude"].isna()]

    lat = r_median(d2[".latitude"], na_rm=True)
    lon = r_median(d2[".longitude"], na_rm=True)

    near = find_near_lat_lon(
        lat, lon, ref=d2, nmax=len(d2), lat_col=".latitude", lon_col=".longitude"
    )
    near = near.rename(columns={"distance_m": "distance.m"})

    # R orders decreasing; order() is stable, so ties keep their prior order.
    order = np.argsort(-near["distance.m"].to_numpy(dtype="float64"), kind="stable")
    near = near.iloc[order].reset_index(drop=True)
    out = near[[".sample_id", ".latitude", ".longitude", "distance.m"]]

    if output is not None and str(output).lower() in ("report", "full.report"):
        return out
    return r_summary_factor(r_cut(out["distance.m"], list(_LATLON_BREAKS)))

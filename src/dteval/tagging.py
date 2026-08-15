"""Diffusion tube pre-processing -- port of ``R/tag.tube.data.R``.

Tagging normalises however the data arrived into the small set of columns the
rest of the package works with (``.start_date``, ``.end_date``, ``.latitude``,
``.longitude``, ``.sample_id``, ``.value``, and the minor ``.date``,
``.month``, ``.year``, ``.location``), while leaving the original columns
untouched.

Argument naming
---------------
R passes overrides through ``...`` with dotted names (``startend.method``,
``latlon.force``, ``date.format``). Python cannot spell those, so the
underscore form is used: ``startend_method``, ``latlon_force``,
``date_format``. The *column* names are unchanged, because those are data.

New columns are appended in the same order R appends them; column order is part
of what the parity suite compares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dteval.handlers import DTEvalError, get_tube_x
from dteval.rcompat import dates as rdates
from dteval.rcompat.coerce import as_character_series, as_matrix_paste
from dteval.rcompat.factor import as_numeric_factor, r_factor

__all__ = [
    "tag_tube",
    "tag_tube_date",
    "tag_tube_lat_lon",
    "tag_tube_location",
    "tag_tube_month",
    "tag_tube_required",
    "tag_tube_sample_id",
    "tag_tube_start_end",
    "tag_tube_value",
    "tag_tube_year",
]


def _opt(kwargs: dict, name: str, default=None):
    return kwargs.get(name, default)


def _round_half_even(x: np.ndarray) -> np.ndarray:
    """R's ``round()`` -- half to even, which is what ``coerceTimeUnit`` uses."""
    return np.rint(np.asarray(x, dtype="float64"))


def _override(kwargs: dict, prefix: str, method, force):
    """Apply R's ``<prefix>.method`` / ``<prefix>.force`` overrides."""
    if f"{prefix}_force" in kwargs:
        force = kwargs[f"{prefix}_force"]
    if f"{prefix}_method" in kwargs:
        method = kwargs[f"{prefix}_method"]
    return method, force


def _as_methods(method, available: list[int], fun_nm: str) -> list[int]:
    if method is None:
        return list(available)
    methods = [method] if np.isscalar(method) else list(method)
    if all(m == -1 for m in methods):
        return list(available)
    bad = [m for m in methods if m not in available]
    if bad:
        raise DTEvalError(
            f"[{fun_nm}]> unknown method(s) '{','.join(str(b) for b in bad)}'"
            f"\n\trecommend one of: {', '.join(str(a) for a in available)}"
            f"\n\t(and maybe check ?{fun_nm}) \n"
        )
    return methods


# --------------------------------------------------------------- tagTube ---


def tag_tube(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run the four main taggers in R's order.

    ``tagTubeStartEnd`` -> ``tagTubeLatLon`` -> ``tagTubeSampleID`` ->
    ``tagTubeValue`` (``tag.tube.data.R:193``). The order matters: sample ids
    are built from the dates and locations the first two produce.
    """
    data = tag_tube_start_end(data, **kwargs)
    data = tag_tube_lat_lon(data, **kwargs)
    data = tag_tube_sample_id(data, **kwargs)
    data = tag_tube_value(data, **kwargs)
    return data


def tag_tube_required(data: pd.DataFrame, required=None, **kwargs) -> pd.DataFrame:
    """Add only the tags named in ``required``.

    A variation on :func:`tag_tube` used throughout the package to avoid
    building tags a given call does not need.
    """
    if required is None:
        return data
    if isinstance(required, str):
        required = [required]
    required = [r for r in required if isinstance(r, str)]

    for item in required:
        if item in (".start_date", ".end_date"):
            data = tag_tube_start_end(data, **kwargs)
        if item == ".date":
            data = data.copy()
            data[".date"] = tag_tube_date(data, **kwargs)[".date"]
        if item == ".month":
            data = data.copy()
            data[".month"] = tag_tube_month(data, **kwargs)[".month"]
        if item == ".year":
            data = data.copy()
            data[".year"] = tag_tube_year(data, **kwargs)[".year"]
        if item in (".latitude", ".longitude"):
            data = tag_tube_lat_lon(data, **kwargs)
        if item == ".location":
            data = data.copy()
            data[".location"] = tag_tube_location(data, **kwargs)[".location"]
        if item == ".sample_id":
            data = tag_tube_sample_id(data, **kwargs)
        if item == ".value":
            data = tag_tube_value(data, **kwargs)
    return data


# ------------------------------------------------------- tagTubeStartEnd ---


def tag_tube_start_end(data: pd.DataFrame, method=-1, force: bool = False, **kwargs):
    """Build ``.start_date`` / ``.end_date``.

    Methods, tried in order when ``method=-1``:

    1. nominal calendar month from month + year columns,
    2. the Defra/LAQM sampling calendar (:func:`dteval.datasets.dt_calendar`),
    3. user-supplied ``start`` / ``end`` columns.
    """
    method, force = _override(kwargs, "startend", method, force)
    funs = {
        1: _start_end_method01,
        2: _start_end_method02,
        3: _start_end_method03,
    }
    methods = _as_methods(method, [1, 2, 3], "tagTubeStartEnd")

    have = ".start_date" in data.columns and ".end_date" in data.columns
    if have and not force:
        return data

    ans = None
    for m in methods:
        try:
            ans = funs[m](data, **kwargs)
            break
        except Exception:  # noqa: BLE001 -- R uses try() and falls through
            ans = None
    if ans is None:
        raise DTEvalError(
            "[tagTubeStartEnd]> failed to match/build start and/or end dates"
            "\n\t(maybe check ?tagTubeStartEnd) \n"
        )
    return ans


def _start_end_method01(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Nominal calendar month: first to last day of the month."""
    month_col = _opt(kwargs, "month", "month")
    year_col = _opt(kwargs, "year", "year_of_measurement")
    if month_col not in data.columns or year_col not in data.columns:
        raise KeyError(month_col)

    months = rdates.parse_month(data[month_col])
    years = pd.to_numeric(pd.Series(data[year_col]), errors="coerce").to_numpy(dtype="float64")
    if np.all(np.isnan(months)) or np.all(np.isnan(years)):
        raise ValueError("no parseable month/year")

    out = data.copy()
    out[".start_date"] = rdates.first_of_month(years, months).to_numpy()
    out[".end_date"] = rdates.end_of_month(years, months).to_numpy()
    return out


def _start_end_method02(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Join onto the Defra/LAQM sampling calendar.

    Note the join *reorders* the data: R uses ``merge.data.table`` here, which
    sorts by the join keys. That row order is part of the output.
    """
    from dteval.datasets import dt_calendar

    month_col = _opt(kwargs, "month", "month")
    year_col = _opt(kwargs, "year", "year_of_measurement")
    if month_col not in data.columns or year_col not in data.columns:
        raise KeyError(month_col)

    cal = dt_calendar()[["year", "month", "start", "end"]].copy()
    cal["year"] = pd.to_numeric(cal["year"], errors="coerce").astype("float64")

    left = data.copy()
    if pd.api.types.is_numeric_dtype(left[month_col].dtype):
        cal["month"] = rdates.parse_month(cal["month"])
    else:
        left[month_col] = as_character_series(left[month_col])
        widths = left[month_col].dropna().map(len)
        widths = widths[widths != 0]
        if len(widths) and (widths == 3).all():
            cal["month"] = cal["month"].str.slice(0, 3)
        else:
            # R's `if` yields NULL here, which deletes the column and makes the
            # subsequent rename fail -- so method 2 is abandoned and method 3
            # gets its turn. Reproduce that by failing.
            raise ValueError("month column is not 3-letter abbreviations")

    cal = cal.rename(
        columns={
            "year": year_col,
            "month": month_col,
            "start": ".start_date",
            "end": ".end_date",
        }
    )
    left["__yr_key"] = pd.to_numeric(left[year_col], errors="coerce").astype("float64")
    cal["__yr_key"] = cal[year_col].astype("float64")
    cal = cal.drop(columns=[year_col])

    out = left.merge(cal, how="left", on=["__yr_key", month_col], sort=True)
    out = out.drop(columns=["__yr_key"]).reset_index(drop=True)

    # data.table's merge puts the join-key columns first, then the remaining
    # left columns, then the right ones. Column order is part of the result.
    key_cols = [year_col, month_col]
    rest = [c for c in data.columns if c not in key_cols]
    added = [c for c in out.columns if c not in key_cols and c not in rest]
    return out[[*key_cols, *rest, *added]]


def _start_end_method03(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Use explicit ``start`` / ``end`` columns."""
    start_col = _opt(kwargs, "start", ".start_date")
    end_col = _opt(kwargs, "end", ".end_date")
    fmt = _opt(kwargs, "date_format", "%Y-%m-%d")
    if start_col not in data.columns or end_col not in data.columns:
        raise KeyError(start_col)

    out = data.copy()
    for src, dest in ((start_col, ".start_date"), (end_col, ".end_date")):
        col = data[src]
        if pd.api.types.is_datetime64_any_dtype(col.dtype):
            out[dest] = col.to_numpy()
        else:
            out[dest] = rdates.parse_date(as_character_series(col), fmt).to_numpy()
    return out


# --------------------------------------------------------- tagTubeLatLon ---


def tag_tube_lat_lon(data: pd.DataFrame, method=-1, force: bool = False, **kwargs):
    """Tag ``.latitude`` / ``.longitude``, assumed WGS84 (EPSG:4326)."""
    lat_col = _opt(kwargs, "lat", "latitude")
    lon_col = _opt(kwargs, "lon", "longitude")
    method, force = _override(kwargs, "latlon", method, force)

    if ".latitude" in data.columns and ".longitude" in data.columns and not force:
        return data

    missing = [
        f"'{col}' ({label})"
        for label, col in (("lat", lat_col), ("lon", lon_col))
        if col not in data.columns
    ]
    if missing:
        raise DTEvalError(
            "[tagTubelatlon]> expected latlon source(s) not in data\n\t"
            f"missing: {','.join(missing)}"
            "\n\t(maybe check data or ?tagTubeLatLon) \n"
        )

    _as_methods(method, [1], "tagTubeLatLon")
    out = data.copy()
    out[".latitude"] = data[lat_col].to_numpy()
    out[".longitude"] = data[lon_col].to_numpy()
    return out


# ------------------------------------------------------- tagTubeSampleID ---


def tag_tube_sample_id(data: pd.DataFrame, method=-1, force: bool = False, **kwargs):
    """Tag ``.sample_id``: co-located, co-timed tubes are replicates.

    R builds the key with ``apply(test, 1, paste, collapse = "-")``, and
    ``apply`` coerces the frame to a character matrix first --
    ``as.matrix.data.frame`` formats numeric columns with ``format()``, *not*
    ``as.character()``. So the key holds ``-1.732780``, not ``-1.73278``.
    The id is then the 1-based factor level index, which makes it depend on
    collation order as well. Both details change the values, not just the
    labels; see docs/parity.md.
    """
    method, force = _override(kwargs, "sampleid", method, force)
    if ".sample_id" in data.columns and not force:
        return data

    data = tag_tube_start_end(data, **kwargs)
    data = tag_tube_lat_lon(data, **kwargs)
    _as_methods(method, [1], "tagTubeSampleID")

    cols = [".latitude", ".longitude", ".start_date", ".end_date"]
    key = as_matrix_paste(data[cols], sep="-")

    out = data.copy()
    out[".sample_id"] = as_numeric_factor(r_factor(key))
    return out


# ---------------------------------------------------------- tagTubeValue ---


def tag_tube_value(data: pd.DataFrame, method=-1, force: bool = False, **kwargs):
    """Tag ``.value`` -- the tube measurement.

    Sources are tried in order; the default prefers the bias-adjusted
    measurement over the raw one.
    """
    value_cols = _opt(kwargs, "value", ["bias_adjusted_measurement", "measurement"])
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    method, force = _override(kwargs, "value", method, force)

    if ".value" in data.columns and not force:
        return data

    _as_methods(method, [1], "tagTubeValue")
    found = None
    for candidate in value_cols:
        if found is None:
            found = get_tube_x(data, candidate)
    if found is None:
        raise DTEvalError(
            "[tagTubeValue]> expected to find one of following: "
            f"\n\t{', '.join(value_cols)}"
            "\n\t(maybe see ?tagTubeValue or rerun with value set?) \n"
        )
    out = data.copy()
    out[".value"] = pd.Series(found).to_numpy()
    return out


# ----------------------------------------------------- minor tags ---------


def tag_tube_date(data: pd.DataFrame, method: int = 2, force: bool = False, **kwargs):
    """Tag ``.date``: 1 = start, 2 = mid-point (default), 3 = end.

    The mid-point lands on a whole day even for odd-length sampling periods.
    R writes it as ``.start_date + ((.end_date - .start_date)/2)``, where the
    right-hand side is a ``difftime``; ``+.Date`` passes that through
    ``coerceTimeUnit``, which applies ``round()`` -- half to **even**. So a
    35-day period gives ``start + 18`` (17.5 rounds up to the even 18) and a
    33-day period gives ``start + 16`` (16.5 rounds down to the even 16).
    """
    method, force = _override(kwargs, "date", method, force)
    if ".date" in data.columns and not force:
        return data

    data = tag_tube_start_end(data, **kwargs)
    if method not in (1, 2, 3):
        raise DTEvalError("[setTubeDate] Unknown method, maybe try one of: 1,2,3")

    start = rdates.as_numeric_date(data[".start_date"])
    end = rdates.as_numeric_date(data[".end_date"])
    if method == 1:
        days = start
    elif method == 2:
        days = start + _round_half_even((end - start) / 2)
    else:
        days = end

    out = data.copy()
    out[".date"] = rdates.date_from_days(days).to_numpy()
    return out


def tag_tube_month(data: pd.DataFrame, method: int = 1, force: bool = False, **kwargs):
    """Tag ``.month`` -- a factor with all twelve ``month.abb`` levels."""
    method, force = _override(kwargs, "month", method, force)
    if ".month" in data.columns and not force:
        return data

    temp = tag_tube_date(data, **kwargs)
    if method != 1:
        raise DTEvalError("[setTubeMonth] Unknown method, maybe try one of: 1")

    labels = rdates.r_format_date(temp[".date"], "%b")
    out = data.copy()
    # Levels are month.abb in calendar order, not the order encountered, and
    # months absent from the data still get a (zero-count) level.
    out[".month"] = r_factor(labels, levels=list(rdates.MONTH_ABB))
    return out


def tag_tube_year(data: pd.DataFrame, method: int = 1, force: bool = False, **kwargs):
    """Tag ``.year`` -- an *ordered* factor of the years present."""
    method, force = _override(kwargs, "year", method, force)
    if ".year" in data.columns and not force:
        return data

    temp = tag_tube_date(data, **kwargs)
    if method != 1:
        raise DTEvalError("[tagTubeYear] Unknown method, maybe try one of: 1")

    labels = rdates.r_format_date(temp[".date"], "%Y")
    from dteval.rcompat.collate import r_sort, r_unique

    levels = r_sort(r_unique(labels))
    out = data.copy()
    out[".year"] = r_factor(labels, levels=levels, ordered=True)
    return out


def tag_tube_location(data: pd.DataFrame, method: int = 1, force: bool = False, **kwargs):
    """Tag ``.location`` as ``{lat,lon}``.

    Built with ``paste``, so the coordinates go through ``as.character`` (15
    significant digits, trailing zeros dropped) -- unlike ``.sample_id``, which
    goes through ``format``. The two therefore render the same number
    differently, and that is faithful to R.
    """
    # R reads location.force/location.method from date.force/date.method here
    # (tag.tube.data.R:839) -- a copy/paste slip upstream, reproduced so the
    # override behaves identically.
    force = kwargs.get("date_force", force) if "location_force" in kwargs else force
    method = kwargs.get("date_method", method) if "location_method" in kwargs else method

    if ".location" in data.columns and not force:
        return data

    data = tag_tube_lat_lon(data, **kwargs)
    if method != 1:
        raise DTEvalError("[setTubeLocation] Unknown method, maybe try one of: 1")

    lat = as_character_series(data[".latitude"]).fillna("NA")
    lon = as_character_series(data[".longitude"]).fillna("NA")
    out = data.copy()
    out[".location"] = ["{" + a + "," + b + "}" for a, b in zip(lat, lon, strict=True)]
    return out


def _unused(*_args: Any) -> None:  # pragma: no cover
    return None

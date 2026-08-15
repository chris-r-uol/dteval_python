"""R ``Date`` semantics.

An R ``Date`` is a **double** counting days since 1970-01-01, not an integer,
so a fractional day is representable and has to be handled rather than assumed
away. Two distinct behaviours matter:

* ``format()`` on a fractional Date **truncates** toward negative infinity
  rather than rounding: 19000.5 and 19000.0 render as the same day, and -0.5
  renders as 1969-12-31.
* ``+.Date`` does not preserve fractions at all. Adding a ``difftime`` goes
  through ``coerceTimeUnit``, which applies ``round()`` -- half to **even**.
  That is why ``tagTubeDate``'s mid-point method lands on a whole day: a 35-day
  sampling period adds 18 (17.5 -> 18) while a 33-day one adds 16 (16.5 -> 16).

We represent Dates as ``datetime64[ns]``, which holds a fractional day exactly,
and implement both behaviours on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.rcompat.rserial import datetime64_to_days, days_to_datetime64

# R's month.abb / month.name under LC_TIME=C.
MONTH_ABB = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
MONTH_NAME = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

__all__ = [
    "MONTH_ABB",
    "MONTH_NAME",
    "as_numeric_date",
    "date_from_days",
    "end_of_month",
    "first_of_month",
    "parse_date",
    "parse_month",
    "r_format_date",
    "floor_date_days",
    "seq_dates",
]


def as_numeric_date(values) -> np.ndarray:
    """R's ``as.numeric(<Date>)`` -- days since 1970-01-01, fraction preserved."""
    return datetime64_to_days(values)


def date_from_days(days) -> pd.Series:
    """R's ``as.Date(<numeric>)`` (origin 1970-01-01), fraction preserved."""
    return pd.Series(days_to_datetime64(np.asarray(days, dtype="float64")))


def floor_date_days(days: np.ndarray) -> np.ndarray:
    """Resolve a possibly-fractional Date to the whole day R's ``format`` uses.

    R **truncates toward negative infinity** rather than rounding: 19000.5 and
    19000.0 both format as 2022-01-08, and -0.5 formats as 1969-12-31. So a
    mid-point ``.date`` of 2022-04-16T12:00 reports as April 16th, not the 17th.
    """
    days = np.asarray(days, dtype="float64")
    out = np.floor(days)
    out[np.isnan(days)] = np.nan
    return out


def _whole_days(values) -> pd.Series:
    """The values as whole-day Timestamps, ready for strftime."""
    days = floor_date_days(as_numeric_date(values))
    return pd.Series(days_to_datetime64(days))


def r_format_date(values, fmt: str = "%Y-%m-%d") -> pd.Series:
    """R's ``format.Date()`` under ``LC_TIME=C``.

    Supports the specifiers DTEval uses: ``%Y %y %m %d %b %B %j``.
    """
    ts = _whole_days(values)
    out = []
    for t in ts:
        if pd.isna(t):
            out.append(pd.NA)
            continue
        s = fmt
        s = s.replace("%Y", f"{t.year:04d}")
        s = s.replace("%y", f"{t.year % 100:02d}")
        s = s.replace("%m", f"{t.month:02d}")
        s = s.replace("%d", f"{t.day:02d}")
        s = s.replace("%B", MONTH_NAME[t.month - 1])
        s = s.replace("%b", MONTH_ABB[t.month - 1])
        s = s.replace("%j", f"{t.dayofyear:03d}")
        out.append(s)
    return pd.Series(out, dtype="object")


def parse_month(values) -> np.ndarray:
    """Month number (1-12) from numeric, 3-letter abbreviation or full name.

    Mirrors ``tagTubeStartEnd_method01``'s sniffing (``tag.tube.data.R:351``):
    numeric months are taken as-is, otherwise all-3-character values are read as
    ``%b`` and anything else as ``%B``. R's ``strptime`` is case-insensitive for
    month names, and the R comment notes it "seems to handle character.month
    case variations" -- so we lower-case before matching.
    """
    vals = list(values)
    out = np.full(len(vals), np.nan)
    abb = {m.lower(): i + 1 for i, m in enumerate(MONTH_ABB)}
    full = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAME)}
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and v != v) or v is pd.NA:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[i] = float(v)
            continue
        key = str(v).strip().lower()
        if key in abb:
            out[i] = abb[key]
        elif key in full:
            out[i] = full[key]
    return out


def parse_date(values, fmt: str = "%Y-%m-%d") -> pd.Series:
    """R's ``as.Date(as.character(x), format = fmt)``.

    Unparseable values become ``NA`` rather than raising, matching R.
    """
    return pd.Series(
        pd.to_datetime(pd.Series(values).astype("object"), format=fmt, errors="coerce")
    )


def end_of_month(year: np.ndarray, month: np.ndarray) -> pd.Series:
    """Last day of the given month.

    This reproduces ``tagTubeStartEnd_method01``'s trick of incrementing the
    ``POSIXlt`` month field and subtracting a day (``tag.tube.data.R:368``),
    including the December wrap into the following year, which ``POSIXlt``
    normalisation handles silently in R.
    """
    year = np.asarray(year, dtype="float64")
    month = np.asarray(month, dtype="float64")
    out = np.full(year.shape, np.nan)
    ok = ~(np.isnan(year) | np.isnan(month))
    yy = year.copy()
    mm = month.copy()
    # month + 1, normalised the way POSIXlt does it
    mm_next = mm + 1
    yy_next = yy + np.where(mm_next > 12, 1, 0)
    mm_next = np.where(mm_next > 12, mm_next - 12, mm_next)
    for i in np.flatnonzero(ok):
        first_next = pd.Timestamp(year=int(yy_next[i]), month=int(mm_next[i]), day=1)
        out[i] = (first_next - pd.Timestamp("1970-01-01")).days - 1
    return date_from_days(out)


def first_of_month(year, month) -> pd.Series:
    """First day of the given month, as an R ``Date``."""
    year = np.asarray(year, dtype="float64")
    month = np.asarray(month, dtype="float64")
    out = np.full(year.shape, np.nan)
    ok = ~(np.isnan(year) | np.isnan(month))
    for i in np.flatnonzero(ok):
        m = int(month[i])
        if not 1 <= m <= 12:
            continue
        ts = pd.Timestamp(year=int(year[i]), month=m, day=1)
        out[i] = (ts - pd.Timestamp("1970-01-01")).days
    return date_from_days(out)


def seq_dates(start: float, end: float) -> np.ndarray:
    """R's ``seq(start, stop)`` over Dates -- inclusive, one-day steps."""
    if np.isnan(start) or np.isnan(end):
        return np.array([], dtype="float64")
    return np.arange(np.floor(start), np.floor(end) + 1, 1.0)

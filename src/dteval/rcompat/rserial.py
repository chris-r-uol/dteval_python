"""Reader for the typed-JSON form that ``parity/serialize.R`` writes.

The same format serves two purposes, so it lives in the shipped package rather
than in ``parity/``:

* the parity harness reads R fixtures with it, and
* the bundled datasets are stored in it, so ``dt.brd`` arrives in Python with
  exactly the classes it had in R (integer vs numeric, Date, factor levels)
  rather than whatever a CSV round-trip would guess.

Encoding rules (mirrored in ``parity/serialize.R``):

* doubles -- ``%.17g`` strings, which round-trip every IEEE double exactly, with
  ``NA`` / ``NaN`` / ``Inf`` / ``-Inf`` as literal tokens. R's ``NA_real_`` and
  ``NaN`` are different values and are kept distinct.
* character / factor / Date -- values plus an explicit list of NA positions,
  because no string sentinel is safe against data that might contain it.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "datetime64_to_days",
    "days_to_datetime64",
    "load_column",
    "load_value",
    "read_rjson",
]

_NA = "NA"
_NAN = "NaN"
_INF = "Inf"
_NEG_INF = "-Inf"


def _parse_double(tok: str) -> float:
    if tok == _NA:
        return np.nan
    if tok == _NAN:
        return math.nan
    if tok == _INF:
        return math.inf
    if tok == _NEG_INF:
        return -math.inf
    return float(tok)


def _parse_int(tok: str):
    return pd.NA if tok == _NA else int(tok)


def _parse_lgl(tok: str):
    if tok == _NA:
        return pd.NA
    return tok == "TRUE"


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _na_positions(col: dict[str, Any]) -> set[int]:
    return set(_as_list(col.get("na")))


def load_column(col: dict[str, Any]) -> pd.Series:
    """Rebuild a pandas Series from one serialised R column."""
    rtype = col["rtype"]
    vals = _as_list(col.get("values"))
    nas = _na_positions(col)

    if rtype == "numeric":
        return pd.Series([_parse_double(v) for v in vals], dtype="float64")
    if rtype == "integer":
        return pd.Series([_parse_int(v) for v in vals], dtype="Int32")
    if rtype == "logical":
        return pd.Series([_parse_lgl(v) for v in vals], dtype="boolean")
    if rtype == "character":
        return pd.Series([pd.NA if i in nas else v for i, v in enumerate(vals)], dtype="object")
    if rtype in ("factor", "ordered"):
        cats = _as_list(col.get("levels"))
        return pd.Series(
            pd.Categorical(
                [None if i in nas else v for i, v in enumerate(vals)],
                categories=cats,
                ordered=(rtype == "ordered"),
            )
        )
    if rtype == "Date":
        # Dates arrive as the underlying R numeric (days since 1970-01-01),
        # which may be fractional -- R's Date is a double. datetime64[ns]
        # represents a fractional day exactly, so no precision is lost.
        days = np.array(
            [np.nan if i in nas else _parse_double(v) for i, v in enumerate(vals)],
            dtype="float64",
        )
        return pd.Series(days_to_datetime64(days))
    if rtype == "POSIXct":
        return pd.Series(
            [pd.NaT if i in nas else pd.Timestamp(v) for i, v in enumerate(vals)],
            dtype="datetime64[ns]",
        )
    raise ValueError(f"unsupported serialised rtype: {rtype!r}")


_NS_PER_DAY = 86_400_000_000_000


def days_to_datetime64(days: np.ndarray) -> np.ndarray:
    """R ``Date`` (days since 1970-01-01, possibly fractional) -> datetime64[ns]."""
    days = np.asarray(days, dtype="float64")
    ns = np.rint(days * _NS_PER_DAY)
    out = np.full(days.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = ~np.isnan(days)
    out[ok] = ns[ok].astype("int64").view("datetime64[ns]")
    return out


def datetime64_to_days(values) -> np.ndarray:
    """datetime64[ns] -> R ``Date`` numeric (days since 1970-01-01)."""
    arr = pd.Series(values).to_numpy(dtype="datetime64[ns]")
    ints = arr.view("int64").astype("float64")
    ints[pd.isna(pd.Series(arr)).to_numpy()] = np.nan
    return ints / _NS_PER_DAY


def load_value(node: dict[str, Any]) -> Any:
    """Rebuild a Python object from a serialised R value."""
    t = node["type"]
    if t == "NULL":
        return None
    if t == "data.frame":
        cols = _as_list(node.get("columns"))
        df = pd.DataFrame({c["name"]: load_column(c) for c in cols})
        if node.get("nrow", len(df)) == 0:
            df = df.iloc[0:0]
        return df.reset_index(drop=True)
    if t == "vector":
        s = load_column(node)
        names = node.get("vnames")
        if names is not None and names != _NA:
            s.index = _as_list(names)
        return s
    if t == "list":
        names = node.get("names")
        values = [load_value(v) for v in _as_list(node.get("values"))]
        if names is None or names == _NA:
            return values
        return dict(zip(_as_list(names), values, strict=False))
    if t == "matrix":
        flat = load_column(node["data"]).to_numpy()
        return flat.reshape(tuple(_as_list(node["dim"])), order="F")  # R fills column-major
    if t == "ggplot":
        return node  # compared structurally, see parity/compare.py
    raise ValueError(f"unsupported serialised type: {t!r}")


def read_rjson(path: str | Path) -> Any:
    """Read a ``.json`` / ``.json.gz`` payload and return its deserialised value."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:  # type: ignore[operator]
        payload = json.load(fh)
    return load_value(payload["value"] if "value" in payload else payload)

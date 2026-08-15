"""R coercion rules for whole vectors.

``as.character`` on a vector is not one rule but several, chosen by class, and
DTEval's tags are built out of the results -- ``.location`` is a pasted
lat/lon string, ``.sample_id`` comes from a pasted location/date key -- so the
choice of rule changes values rather than appearance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.rcompat import numfmt

__all__ = ["as_character_series", "as_matrix_paste", "as_numeric_series", "is_na"]


def is_na(values) -> np.ndarray:
    """R's ``is.na()`` -- true for NA, NaN and NaT alike."""
    return pd.isna(pd.Series(values)).to_numpy()


def as_character_series(values, use_format: bool = False) -> pd.Series:
    """R's ``as.character()`` for a vector.

    ``use_format=True`` switches to ``format()`` semantics instead -- common
    width and ``getOption("digits")`` significant digits, trailing zeros kept.
    That is what ``as.matrix.data.frame`` applies when ``apply()`` coerces a
    frame to a character matrix, and it produces different strings from
    ``as.character`` (``-1.732780`` vs ``-1.73278``).
    """
    s = pd.Series(values)
    na = pd.isna(s).to_numpy()

    if isinstance(s.dtype, pd.CategoricalDtype):
        out = s.astype(object).map(lambda v: None if pd.isna(v) else str(v))
    elif pd.api.types.is_datetime64_any_dtype(s):
        from dteval.rcompat.dates import r_format_date

        out = r_format_date(s, "%Y-%m-%d")
    elif pd.api.types.is_bool_dtype(s.dtype):
        out = s.map(lambda v: None if pd.isna(v) else ("TRUE" if v else "FALSE"))
    elif pd.api.types.is_integer_dtype(s.dtype):
        out = s.map(lambda v: None if pd.isna(v) else str(int(v)))
    elif pd.api.types.is_float_dtype(s.dtype):
        if use_format:
            out = pd.Series(numfmt.r_format_numeric(s.to_numpy(dtype="float64")), index=s.index)
        else:
            out = s.map(lambda v: None if pd.isna(v) else numfmt.as_character(float(v)))
    else:
        out = s.map(lambda v: None if pd.isna(v) else str(v))

    out = pd.Series(list(out), index=s.index, dtype="object")
    out[na] = pd.NA
    return out


def as_matrix_paste(df: pd.DataFrame, sep: str) -> list[str]:
    """``apply(df, 1, paste, collapse = sep)`` -- with R's coercion rules.

    ``apply`` first turns the frame into a character matrix via
    ``as.matrix.data.frame``, which is *not* ``as.character`` column by column:

    * a factor uses ``as.vector`` -- its labels;
    * anything else numeric goes through ``format()``, so a column shares one
      width and one decimal count (``-1.732780``, not ``-1.73278``).

    DTEval builds two different keys this way -- ``.sample_id``
    (``tag.tube.data.R:571``) and the boxplot's ``..group``
    (``tube.plots.R:937``) -- and both are values, not labels.
    """
    parts: list[list[str]] = []
    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            parts.append(list(as_character_series(series).fillna("NA")))
        elif pd.api.types.is_float_dtype(series.dtype):
            parts.append(numfmt.r_format_numeric(series.to_numpy(dtype="float64")))
        else:
            parts.append(list(as_character_series(series, use_format=True).fillna("NA")))
    return [sep.join(row) for row in zip(*parts, strict=True)]


def as_numeric_series(values) -> pd.Series:
    """R's ``as.numeric()``.

    A ``factor`` yields its 1-based level index, not the label parsed as a
    number -- the behaviour ``tagTubeSampleID`` relies on.
    """
    s = pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        from dteval.rcompat.factor import as_numeric_factor

        return pd.Series(as_numeric_factor(s), index=s.index, dtype="float64")
    if pd.api.types.is_datetime64_any_dtype(s):
        from dteval.rcompat.dates import as_numeric_date

        return pd.Series(as_numeric_date(s), index=s.index, dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")

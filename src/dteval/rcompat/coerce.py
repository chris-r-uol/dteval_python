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

    ``use_format`` is accepted for call-site compatibility but no longer
    switches behaviour: the ``format()`` path existed only so ``.sample_id``'s
    pasted key matched R byte for byte, and that key is now built from a tuple.
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
        out = s.map(lambda v: None if pd.isna(v) else numfmt.as_character(float(v)))
    else:
        out = s.map(lambda v: None if pd.isna(v) else str(v))

    out = pd.Series(list(out), index=s.index, dtype="object")
    out[na] = pd.NA
    return out


def as_matrix_paste(df: pd.DataFrame, sep: str) -> list[str]:
    """Join each row's values into one string, as ``apply(df, 1, paste)`` does.

    Used for the boxplot's ``..group`` key (``tube.plots.R:937``), where the
    string only has to separate groups consistently -- it is never shown.
    """
    parts = [list(as_character_series(df[col]).fillna("NA")) for col in df.columns]
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

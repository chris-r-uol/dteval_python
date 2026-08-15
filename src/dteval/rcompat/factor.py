"""R ``factor`` semantics.

``factor(x)`` sets its levels to ``sort(unique(x))`` with NA dropped, and
``as.numeric(<factor>)`` returns the 1-based level index. DTEval leans on that
combination in ``tagTubeSampleID`` (``tag.tube.data.R:571``):

    test <- as.factor(apply(test, 1, paste, collapse = "-"))
    data$.sample_id <- as.numeric(test)

so a sample's *integer id* is its position in the collation-sorted set of
location/date keys. Level order is therefore a value, not a presentation
detail -- see :mod:`dteval.rcompat.collate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.rcompat.collate import r_sort, r_unique

__all__ = [
    "as_numeric_factor",
    "make_names",
    "r_factor",
    "r_summary_factor",
    "summary_factor_frame",
]


def r_factor(values, levels=None, ordered: bool = False) -> pd.Series:
    """R's ``factor()`` -- levels default to ``sort(unique(x))``, NA excluded."""
    vals = pd.Series(values).astype("object")
    if levels is None:
        levels = r_sort(r_unique(vals))
    levels = [str(v) for v in levels]
    as_str = vals.map(lambda v: pd.NA if _is_na(v) else str(v))
    return pd.Series(
        pd.Categorical(as_str, categories=levels, ordered=ordered), index=vals.index
    )


def _is_na(v) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    return isinstance(v, float) and v != v


def as_numeric_factor(values) -> np.ndarray:
    """R's ``as.numeric(<factor>)`` -- the 1-based level index, NA for missing."""
    s = pd.Series(values)
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = r_factor(s)
    codes = s.cat.codes.to_numpy().astype("float64")
    codes[codes < 0] = np.nan
    return codes + 1.0


def r_summary_factor(values) -> pd.Series:
    """R's ``summary(<factor>)`` -- counts per level, in level order.

    An ``NA's`` entry is appended when any value is missing, as R does.
    """
    s = pd.Series(values)
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = r_factor(s)
    counts = s.value_counts(dropna=True, sort=False)
    counts = counts.reindex(list(s.cat.categories), fill_value=0)
    out = pd.Series(counts.to_numpy().astype("int64"), index=list(s.cat.categories))
    n_na = int(s.isna().sum())
    if n_na:
        out["NA's"] = n_na
    return out


def make_names(names) -> list[str]:
    """R's ``make.names()`` for the cases ``data.frame()`` hits.

    Relevant because ``tubeSummarySample`` returns
    ``data.frame(t(summary(factor(out$n))))`` (``tube.summaries.R:178``): the
    factor levels are the numbers 1, 2, 3, and ``data.frame`` renames them to
    the syntactically valid ``X1``, ``X2``, ``X3``.
    """
    out = []
    for nm in names:
        s = str(nm)
        s = "".join(ch if (ch.isalnum() or ch in "._") else "." for ch in s)
        if not s or s[0].isdigit() or (s[0] == "." and len(s) > 1 and s[1].isdigit()):
            s = "X" + s
        out.append(s)
    return out


def summary_factor_frame(values) -> pd.DataFrame:
    """R's ``data.frame(t(summary(factor(x))))`` -- a one-row count table."""
    counts = r_summary_factor(values)
    cols = make_names(counts.index)
    return pd.DataFrame([counts.to_numpy().astype("int64")], columns=cols).astype("Int32")

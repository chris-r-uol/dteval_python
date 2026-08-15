"""R's merge semantics.

``merge.data.frame`` differs from ``pandas.merge`` in three ways that all show
up in DTEval's output, so they are reproduced here once rather than at each
call site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["r_merge"]


def r_merge(
    x: pd.DataFrame,
    y: pd.DataFrame,
    by: list[str] | None = None,
    how: str = "inner",
) -> pd.DataFrame:
    """Base R's ``merge(x, y)``.

    Three behaviours pandas does not share:

    1. **Join columns are chosen by position in x -- but only when inferred.**
       Called without ``by=``, R uses ``intersect(names(x), names(y))``, which
       is ordered by where the columns sit in ``x``, not by the order the
       caller built them in. Called *with* ``by=``, R keeps the order given.
       DTEval relies on both: ``testTubePrecision`` merges without ``by`` (so
       ``.year`` lands before ``.cut``), while ``tubeInXYPolygon`` passes
       ``by = c(lon, lat)`` and gets longitude first.
    2. **Join columns come first** in the result, then x's remaining columns,
       then y's.
    3. **The result is sorted on the join key as text.** ``merge.data.frame``
       pastes the by-columns into one ``"\\r"``-separated string and orders on
       that, so a numeric key sorts lexicographically: ``.sample_id`` runs
       1, 1151, 1199, ... and 48 lands after 1425, not second.

    Getting (3) wrong reorders every row of ``testTubePrecision``'s output;
    getting (1) wrong puts ``.year`` on the wrong side of ``.cut``.
    """
    from dteval.rcompat.coerce import as_character_series

    if by is None:
        shared = set(x.columns) & set(y.columns)
        by = [c for c in x.columns if c in shared]
    else:
        by = list(by)
    if not by:
        raise ValueError("no columns in common to merge on")

    out = x.merge(y, on=by, how=how)
    rest_x = [c for c in x.columns if c not in by]
    rest_y = [c for c in y.columns if c not in by]
    out = out[[*by, *rest_x, *rest_y]]

    key = None
    for col in by:
        part = as_character_series(out[col]).fillna("NA")
        key = part if key is None else key + "\r" + part
    order = np.argsort(np.asarray(key, dtype=object), kind="stable")
    return out.iloc[order].reset_index(drop=True)

"""R's ordering semantics under the pinned ``LC_COLLATE=C`` locale.

Why this is not just ``sorted()``
---------------------------------
R's ``sort()``, ``order()`` and ``factor()`` level construction all go through
locale collation. Under a UK/US locale, glibc/ICU collation largely ignores
punctuation at the primary strength, so ``"53.7--1.7"`` and ``"537-17"`` can
compare differently than their bytes suggest. DTEval builds ``.sample_id`` as
``as.numeric(factor(paste(lat, lon, start, end, sep = "-")))`` -- the *integer
sample ids therefore depend on collation order*. Row order of
``testTubePrecision`` / ``testTubeAccuracy`` output likewise comes from
``sort(unique(data$.cut))``.

So the reference has to state a locale rather than inherit one. The parity
harness pins ``LC_COLLATE=C``, under which R compares raw bytes with
``strcmp``. UTF-8 preserves code-point order, so Python's default string
comparison reproduces C collation exactly -- which is why these functions look
thin. They exist to make the assumption explicit and greppable, not because the
implementation is clever.

See ``docs/parity.md`` for the consequence: a user running R under a different
locale can get different ``.sample_id`` numbering than this port produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["r_order", "r_sort", "r_unique", "sort_key"]


def sort_key(value):
    """Collation key for one value under ``LC_COLLATE=C``."""
    return value


def _is_na(v) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    return isinstance(v, float) and v != v


def r_unique(values) -> list:
    """R's ``unique()`` -- first occurrence wins, NA is kept and appears once."""
    out: list = []
    seen: set = set()
    seen_na = False
    for v in _to_list(values):
        if _is_na(v):
            if not seen_na:
                seen_na = True
                out.append(v)
            continue
        key = v
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def r_sort(values, decreasing: bool = False) -> list:
    """R's ``sort()`` -- ascending, **NA dropped** (R's ``na.last = NA`` default)."""
    vals = [v for v in _to_list(values) if not _is_na(v)]
    return sorted(vals, key=sort_key, reverse=decreasing)


def r_order(values, decreasing: bool = False) -> np.ndarray:
    """R's ``order()`` -- 0-based here; NA sorts last, as R's ``na.last = TRUE``.

    Ties keep their original relative order, matching R's stable ordering.
    """
    vals = _to_list(values)
    present = [(sort_key(v), i) for i, v in enumerate(vals) if not _is_na(v)]
    missing = [i for i, v in enumerate(vals) if _is_na(v)]
    present.sort(key=lambda t: t[0], reverse=decreasing)
    return np.array([i for _, i in present] + missing, dtype="int64")


def _to_list(values) -> list:
    if isinstance(values, pd.Series):
        if isinstance(values.dtype, pd.CategoricalDtype):
            return list(values.astype(object))
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)

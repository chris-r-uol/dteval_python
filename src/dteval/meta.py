"""Diffusion tube meta-data handling -- port of ``R/misc.tube.meta.R``.

Site meta-data (name, coordinates, classification) is often recorded
inconsistently: present on some rows and blank on others, or spelled two ways
for the same site. These functions extract it, fill the gaps, and flag the
cases that cannot be resolved automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.handlers import DTEvalError, check_tube_data
from dteval.rcompat.collate import r_sort, r_unique
from dteval.rcompat.merge import r_merge
from dteval.tagging import tag_tube, tag_tube_date, tag_tube_location, tag_tube_required

__all__ = [
    "add_tube_meta",
    "check_tube_meta",
    "extract_and_add_tube_meta",
    "extract_tube_meta",
    "pad_tube_meta",
    "repair_tube_meta",
]


def add_tube_meta(
    data: pd.DataFrame, ref: pd.DataFrame | None = None, by: str | list[str] | None = None, **kwargs
) -> pd.DataFrame:
    """Port of ``addTubeMeta`` -- join a meta-data table onto the tube data.

    Columns present in both are taken from ``ref``: the point is to replace
    patchy values, so the reference wins.
    """
    if ref is None:
        return data
    if by is None:
        if ".sample_id" not in data.columns:
            raise DTEvalError("[addTubeMeta] Sorry, need a valid 'by'")
        by = ".sample_id"
    by_cols = [by] if isinstance(by, str) else list(by)

    data = pd.DataFrame(data)
    clashes = [c for c in data.columns if c in ref.columns and c not in by_cols]
    data = data[[c for c in data.columns if c not in clashes]]
    return r_merge(data, ref, by=by_cols)


def check_tube_meta(
    data: pd.DataFrame,
    x: str | None = None,
    by: str | None = None,
    output: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``checkTubeMeta`` -- how many distinct values of ``x`` per ``by``.

    A well-behaved meta-data column has exactly one value per site; anything
    higher is worth a look. Sorted worst-first.
    """
    if x is None or by is None:
        raise DTEvalError("[checkTubeMeta] need both 'x' and 'by'")

    d = tag_tube(data, **kwargs)
    d = check_tube_data(d, x, if_err="stop<<checkTubeMeta>>x")
    if by == ".location":
        d = tag_tube_location(d, **kwargs)
    if by == ".date":
        d = tag_tube_date(d, **kwargs)

    def stat(values):
        uniq = r_unique(values)
        return {
            "count": len(uniq),
            "options": "|".join(f"'{v}'" for v in r_sort(uniq)),
        }

    out = calc_tube_stat(d, x, by=by, stat=stat)
    out = out.sort_values(f"{x}.count", ascending=False, kind="stable").reset_index(drop=True)

    if output is not None and str(output).lower() in ("report", "full.report"):
        return out
    return calc_tube_stat(
        out, tube=by, by=[f"{x}.count", f"{x}.options"], stat=lambda v: len(v)
    )


def extract_tube_meta(
    data: pd.DataFrame,
    x: str | list[str] | None = None,
    by: str | None = None,
    **kwargs,
) -> pd.DataFrame | None:
    """Port of ``extractTubeMeta`` -- pull out the columns that behave like meta-data.

    A column qualifies if it has exactly one non-missing value for every
    ``by`` group. Returns ``None`` when nothing qualifies.
    """
    data = tag_tube_required(data, required=[*( [x] if isinstance(x, str) else (x or []) ),
                                             *([by] if by else [])], **kwargs)
    if by is None:
        if ".sample_id" not in data.columns:
            raise DTEvalError("[padTubeMeta] Sorry, need a valid 'by'")
        by = ".sample_id"
    cols = [x] if isinstance(x, str) else (list(x) if x is not None else
                                           [c for c in data.columns if c != by])

    counts = calc_tube_stat(
        data, cols, by=by, stat=lambda v: len(r_unique(pd.Series(v).dropna()))
    )
    keep = [c for c in cols if (counts[c] == 1).all()]
    if not keep:
        return None

    out = calc_tube_stat(
        data, keep, by=by, stat=lambda v: _first_present(v)
    )
    # calcTubeStat loses the source dtypes; R restores them explicitly.
    from dteval.calc import _restore_dtype

    for col in out.columns:
        if col in data.columns:
            out[col] = _restore_dtype(out[col], data[col])
    return out


def _first_present(values) -> Any:
    """R's ``unique(x[!is.na(x)])[1]`` -- the first non-missing distinct value."""
    present = pd.Series(values).dropna()
    uniq = r_unique(present)
    return uniq[0] if uniq else np.nan


def extract_and_add_tube_meta(
    data: pd.DataFrame,
    x: str | list[str] | None = None,
    by: str | list[str] | None = None,
    ref: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``extractAndAddTubeMeta`` -- extract meta-data from ``ref``, then join it.

    Applied one ``by`` term at a time, as R does, so later terms see the
    columns earlier ones added.
    """
    by_cols = [by] if isinstance(by, str) else list(by or [])
    for key in by_cols:
        meta = extract_tube_meta(ref, x=x, by=key, **kwargs)
        if meta is not None:
            data = add_tube_meta(data, ref=meta, by=key, **kwargs)
    return data


def pad_tube_meta(
    data: pd.DataFrame,
    x: str | None = None,
    by: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``padTubeMeta`` -- fill a patchy column from the rest of its group.

    Every row in a group takes the group's first non-missing value of ``x``,
    which is how a coordinate recorded on only one of three replicates gets
    onto all of them.
    """
    if x is None:
        return data
    if by is None:
        if ".sample_id" not in data.columns:
            raise DTEvalError("[padTubeMeta] Sorry, need a valid 'by'")
        by = ".sample_id"

    frame = check_tube_data(pd.DataFrame(data), [x, by], if_err="stop")[[x, by]]
    filled = (
        frame.groupby(by, sort=False, dropna=False, observed=True)[x]
        .apply(_first_present)
        .reset_index()
    )

    out = pd.DataFrame(data)
    out = out[[c for c in out.columns if c != x]]
    return r_merge(out, filled, by=[by])


def repair_tube_meta(
    data: pd.DataFrame,
    x: str | None = None,
    by: str | None = None,
    options: list[Any] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``repairTubeMeta`` -- resolve inconsistent values against a whitelist.

    Within each ``by`` group, if exactly one of the values found is in
    ``options`` it replaces the rest. If several are, the group is left alone
    but every value is suffixed ``.SUSPECT`` so it can be found later.
    """
    if options is None:
        raise DTEvalError("[repairTubeMeta] need 'options' to work with...")
    if x is None or by is None:
        raise DTEvalError("[repairTubeMeta] need both 'x' and 'by'")

    d = tag_tube(data, **kwargs)
    d = check_tube_data(d, x, if_err="stop<<checkTubeMeta>>x")
    if by == ".location":
        d = tag_tube_location(d, **kwargs)
    if by == ".date":
        d = tag_tube_date(d, **kwargs)

    d = d.copy()
    repaired = suspect = 0
    for key in r_unique(d[by]):
        rows = d[by] == key
        found = r_sort(r_unique(d.loc[rows, x]))
        valid = [o for o in options if o in found]

        if len(valid) == 1 and len(found) > 1:
            d.loc[rows, x] = valid[0]
            repaired += 1
        elif len(valid) > 1:
            d.loc[rows, x] = d.loc[rows, x].astype(str) + ".SUSPECT"
            suspect += 1

    if repaired > 1 or suspect > 1:
        print(
            f"[repairTubeMeta] {repaired} {x} repair(s) made; "
            f"{suspect} suspect(s) subsets identified."
        )
    return d

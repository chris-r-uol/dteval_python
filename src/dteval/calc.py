"""Common diffusion tube calculations -- port of ``R/calc.tube.R``.

``calc_tube_stat`` is the grouped-summary workhorse most other functions route
through, so its output shape has to be exact: column order, group order, and
the ``<column>.<stat>`` naming that comes out of R's
``as.list(unlist(lapply(.SD, stat)))``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from dteval.handlers import DTEvalError, check_tube_data
from dteval.rcompat.stats import r_mean
from dteval.tagging import tag_tube_required

__all__ = ["calc_tube_stat"]


def _default_stat(x) -> dict[str, float]:
    return {"mean": r_mean(x, na_rm=True)}


def calc_tube_stat(
    data: pd.DataFrame,
    tube: str | list[str] = ".value",
    by: str | list[str] | None = None,
    stat: Callable[[Any], Any] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Grouped summary statistics.

    ``stat`` is a callable taking the group's values and returning either a
    scalar or a mapping of ``{name: value}``. The output column for a mapping
    is ``"<tube>.<name>"``; for a bare scalar it is just ``"<tube>"`` -- R gets
    that naming from ``unlist()``, and code such as ``extractTubeMeta`` indexes
    the result by the plain column name, so the distinction matters.

    Groups come out sorted by ``by``, because R sorts the table with
    ``setorderv`` before grouping (``calc.tube.R:140``).
    """
    tube_cols = [tube] if isinstance(tube, str) else list(tube)
    by_cols = [] if by is None else ([by] if isinstance(by, str) else list(by))

    required = [*tube_cols, *by_cols]
    d2 = tag_tube_required(data, required=required, **kwargs)
    d2 = check_tube_data(d2, tube_cols, if_err="stop<<calcTubeStat>>tube")
    if by_cols:
        d2 = check_tube_data(d2, by_cols, if_err="stop<<calcTubeStat>>by")

    by_cols = _split_by_on_commas(by_cols, d2)
    fn = _default_stat if stat is None else stat

    if not by_cols:
        row: dict[str, Any] = {}
        for col in tube_cols:
            _emit(row, col, fn(d2[col]))
        return pd.DataFrame([_unlist_coerce(row)])

    # data.table's setorderv sorts ascending with NA *first* (na.last = FALSE),
    # which differs from base R's sort() dropping NA -- and the resulting order
    # is the group order in the output.
    ordered = _sort_by(d2, by_cols)

    keys = [ordered[c] for c in by_cols]
    grouper = pd.MultiIndex.from_arrays(keys) if len(keys) > 1 else keys[0]
    groups = ordered.groupby(grouper, sort=False, dropna=False, observed=True)

    records: list[dict[str, Any]] = []
    for key, chunk in groups:
        row = {}
        key_vals = key if isinstance(key, tuple) else (key,)
        for name, value in zip(by_cols, key_vals, strict=True):
            row[name] = value
        stats: dict[str, Any] = {}
        for col in tube_cols:
            _emit(stats, col, fn(chunk[col]))
        row.update(_unlist_coerce(stats))
        records.append(row)

    out = pd.DataFrame.from_records(records)
    # Restore the by-columns' original dtypes: going through a groupby key can
    # turn a categorical or Date into object.
    for name in by_cols:
        out[name] = _restore_dtype(out[name], d2[name])
    return out


def _split_by_on_commas(by_cols: list[str], data: pd.DataFrame) -> list[str]:
    """Reproduce ``data.table``'s comma handling for a character ``by``.

    data.table treats a length-1 ``by`` string as a comma-separated list of
    column names, and rejects a longer vector whose elements contain commas.
    So ``by = "paste(year, month)"`` -- which ``checkTubeData`` has just
    evaluated quite happily into a column of that literal name -- is then split
    into ``paste(year`` and ``month)`` and fails.

    The R source flags this as a known wart (``calc.tube.R:87``). It is
    reproduced rather than fixed, because a `by` expression that works here and
    errors in R would be a divergence.
    """
    if not by_cols:
        return by_cols
    if len(by_cols) == 1:
        if "," not in by_cols[0]:
            return by_cols
        parts = [p.strip() for p in by_cols[0].split(",")]
        missing = [p for p in parts if p not in data.columns]
        if missing:
            raise DTEvalError(
                f"object '{missing[0]}' not found\n"
                "\t(a 'by' term may not contain a comma: it is read as a list of "
                "column names -- see calc.tube.R:87)"
            )
        return parts
    offenders = [c for c in by_cols if "," in c]
    if offenders:
        raise DTEvalError(
            f"'by' is a character vector length {len(by_cols)} but one or more items "
            f"include a comma: {offenders}"
        )
    return by_cols


def _unlist_coerce(stats: dict[str, Any]) -> dict[str, Any]:
    """Reproduce ``unlist()``'s type coercion across a stat's outputs.

    R builds the result with ``as.list(unlist(lapply(.SD, stat)))``, and
    ``unlist`` collapses to a single type: if any element is a character, they
    all become characters. So a stat returning both a count and a label emits
    ``"3"``, not ``3`` -- as ``checkTubeMeta`` does.
    """
    if not any(isinstance(v, str) for v in stats.values()):
        return stats

    from dteval.rcompat.numfmt import as_character

    out = {}
    for name, value in stats.items():
        if isinstance(value, str) or value is None:
            out[name] = value
        elif isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[name] = as_character(
                bool(value) if isinstance(value, (bool, np.bool_)) else value
            )
        else:
            out[name] = str(value)
    return out


def _emit(row: dict[str, Any], col: str, value: Any) -> None:
    """Name the output column(s) the way R's ``unlist()`` does."""
    if isinstance(value, dict):
        for stat_name, v in value.items():
            row[f"{col}.{stat_name}" if stat_name else col] = v
        return
    if isinstance(value, pd.Series):
        if value.index.dtype == object and len(value) and all(isinstance(i, str) for i in value.index):
            for stat_name, v in value.items():
                row[f"{col}.{stat_name}"] = v
            return
        value = value.iloc[0] if len(value) == 1 else list(value)
    row[col] = value


def _sort_by(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    """R's ``data.table::setorderv`` -- ascending, NA first, C-locale strings."""
    frame = pd.DataFrame(index=df.index)
    for i, col in enumerate(by_cols):
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            frame[f"k{i}"] = s.cat.codes.replace(-1, np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s.dtype):
            frame[f"k{i}"] = s.astype("int64").where(~s.isna(), np.nan)
        else:
            frame[f"k{i}"] = s
    order = frame.sort_values(
        list(frame.columns), ascending=True, na_position="first", kind="stable"
    ).index
    return df.loc[order]


def _restore_dtype(values: pd.Series, template: pd.Series) -> pd.Series:
    if isinstance(template.dtype, pd.CategoricalDtype):
        return pd.Series(
            pd.Categorical(
                values.astype(object),
                categories=template.cat.categories,
                ordered=template.cat.ordered,
            ),
            index=values.index,
        )
    try:
        return values.astype(template.dtype)
    except (TypeError, ValueError):
        return values

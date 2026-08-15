"""Annual coverage metrics -- port of ``R/misc.tube.annual.R``.

``tube_annual_cover`` reports how much of each year a location was actually
sampled, independent of co-located replicates or site renaming, by counting the
distinct days covered by its sampling periods. ``tube_annual_test`` answers the
narrower question of which locations pass a test in *every* year of a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.handlers import DTEvalError, check_tube_data
from dteval.rcompat.collate import r_sort, r_unique
from dteval.rcompat.dates import as_numeric_date
from dteval.rcompat.merge import r_merge
from dteval.tagging import tag_tube_required, tag_tube_start_end, tag_tube_year

__all__ = ["tube_annual_cover", "tube_annual_test"]


def tube_annual_cover(
    data: pd.DataFrame,
    tube: str = ".value",
    by: str = ".location",
    output: str | list[str] = ("n", "pc"),
    rename: str | list[str] | None = None,
    meta: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``tubeAnnualCover``.

    Adds ``.annual.n`` -- days with at least one measurement at that location
    in that year -- and ``.annual.pc``, that as a percentage of the year.

    Note ``.annual.pc`` divides by **365 regardless of leap years**. That is
    upstream's behaviour (``misc.tube.annual.R:189`` even flags it in a
    comment), and it is reproduced rather than corrected.
    """
    outputs = [output] if isinstance(output, str) else list(output)
    if not all(o in ("n", "pc") for o in outputs):
        raise DTEvalError("[tubeAnnualCount] bad output requested...")
    wanted = [f".annual.{o}" for o in outputs]

    renames = None
    if rename is not None:
        renames = [rename] if isinstance(rename, str) else list(rename)
        if len(renames) != len(wanted):
            raise DTEvalError("[tubeAnnualCount] output/rename mismatch, lengths differ...")

    d2 = tag_tube_required(data, required=[tube, by], **kwargs)
    d2 = tag_tube_start_end(d2)

    temp = calc_tube_stat(d2, tube, by=[".start_date", ".end_date", ".year", by])

    records = []
    # data.table `by=` groups in order of first appearance in the (already
    # sorted) table produced by calcTubeStat.
    for key, chunk in temp.groupby([".year", by], sort=False, dropna=False, observed=True):
        records.append(
            {
                ".year": key[0],
                by: key[1],
                ".annual.n": _days_covered(chunk[".start_date"], chunk[".end_date"]),
            }
        )
    ans = pd.DataFrame.from_records(records)
    ans[".annual.n"] = ans[".annual.n"].astype("Int32")
    ans[".annual.pc"] = ans[".annual.n"].astype("float64") / 365 * 100

    for column in (".annual.n", ".annual.pc"):
        if column not in wanted:
            ans = ans.drop(columns=[column])
    if renames is not None:
        ans = ans.rename(columns=dict(zip(wanted, renames, strict=True)))

    from dteval.calc import _restore_dtype

    ans[".year"] = _restore_dtype(ans[".year"], temp[".year"])
    ans[by] = _restore_dtype(ans[by], temp[by])

    if meta:
        return ans

    d2 = tag_tube_year(d2)
    drop = renames if renames is not None else wanted
    d2 = d2[[c for c in d2.columns if c not in drop]]
    return r_merge(d2, ans, by=[".year", ".location"])


def _days_covered(starts, ends) -> int:
    """Distinct days spanned by the sampling periods.

    R builds ``seq(start, end)`` per row, concatenates, and takes
    ``length(sort(unique(.)))`` -- so overlapping periods are counted once and
    both endpoints are inclusive.
    """
    start_days = as_numeric_date(starts)
    end_days = as_numeric_date(ends)
    covered: set[int] = set()
    for lo, hi in zip(start_days, end_days, strict=True):
        if np.isnan(lo) or np.isnan(hi):
            continue
        covered.update(range(int(np.floor(lo)), int(np.floor(hi)) + 1))
    return len(covered)


def tube_annual_test(
    data: pd.DataFrame,
    test: str | None = None,
    by: str = ".location",
    years: list | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``tubeAnnualTest`` -- which locations pass ``test`` in every year.

    ``test`` names a logical column (or an expression producing one, e.g.
    ``".annual.pc > 75"``). A ``by`` group passes only if the rows where the
    test holds cover *all* the years in ``years`` -- every year present in the
    data, unless given.

    The flag lands in a column named ``<test>.<first year>.<last year>``, or
    whatever ``rename`` says.
    """
    d2 = check_tube_data(data, test, if_err="stop<<tubeAnnualTest>>test")
    d2 = tag_tube_required(d2, required=[by, ".year"], **kwargs)

    if years is None:
        years = list(r_unique(tag_tube_year(d2)[".year"]))
    years = r_sort([str(y) for y in years])
    name = kwargs.get("rename")
    name = name[0] if isinstance(name, (list, tuple)) else name
    if name is None:
        name = f"{test}.{years[0]}.{years[-1]}"

    passing = d2[d2[test].astype(bool)]
    by_cols = [by] if isinstance(by, str) else list(by)
    covered = (
        passing.groupby(by_cols, sort=False, dropna=False, observed=True)[".year"]
        .apply(lambda ys: set(years) <= set(ys.astype(str)))
        .reset_index(name=name)
    )

    # A row belongs to a passing group when it matches on every `by` column.
    hits = pd.Series(0, index=d2.index)
    good = covered[covered[name]]
    for column in by_cols:
        hits = hits + d2[column].isin(r_unique(good[column])).astype(int)

    out = d2[[c for c in d2.columns if c not in (test, name)]].copy()
    out[name] = hits == len(by_cols)
    return out.reset_index(drop=True)

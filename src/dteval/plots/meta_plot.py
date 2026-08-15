"""Meta-data completeness plot -- port of ``R/test.tube.meta.R``.

``test_tube_meta`` shows, for each column, how consistently it is populated at
four levels of aggregation: over the whole data set, per sample, per date and
per location. It is a data-quality triage tool -- a column that is complete
per-location but patchy per-sample is a padding job, one that varies within a
sample is a real inconsistency.

Note this is a *library* function whose name begins with ``test_``. Reach it as
``dteval.test_tube_meta(...)`` rather than importing the name, or pytest will
try to collect it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.plots.ggshell import LayerSpec
from dteval.plots.tube_plot import TubePlot
from dteval.rcompat.factor import r_factor
from dteval.rcompat.palette import grey
from dteval.tagging import tag_tube, tag_tube_date, tag_tube_location

__all__ = ["test_tube_meta"]

#: Aggregation levels, in the order R stacks them.
_LEVELS = (("..all", "all.data"), ("..sample", "sample"), ("..date", "date"),
           ("..location", "location"))

#: R reverses this for the y axis so all.data sits at the top of the chart.
_REF_ORDER = ("all.data", "sample", "location", "date")

#: plot.type 1's fill scale -- ggplot2's four-hue default, spelled out in the
#: R source rather than computed.
_HUE_COLOURS = ("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")

#: plot.type 2's fill scale, in R's palette order: grey (all missing), green
#: (one value), amber (one value plus missing), red (genuinely inconsistent).
_STATUS_ORDER = ("red", "amber", "green", "grey")


def test_tube_meta(
    data: pd.DataFrame,
    x: str | list[str] | None = None,
    by: str | None = None,
    plot_type: int = 2,
    palette: list[str] | None = None,
    **kwargs,
) -> TubePlot:
    """Port of ``testTubeMeta``.

    ``plot_type=1`` shows, per level, the percentage of rows where a column is
    single-valued -- one bar per level. ``plot_type=2`` (default) keeps every
    group as its own stacked segment and colours it by status, so a column that
    is fine at most sites but broken at a few reads as a part-coloured bar
    rather than an averaged-away number.
    """
    d = tag_tube(data, **kwargs)
    columns = list(d.columns) if x is None else ([x] if isinstance(x, str) else list(x))

    d = d.copy()
    d["..location"] = tag_tube_location(d, **kwargs)[".location"]
    d["..date"] = tag_tube_date(d, **kwargs)[".date"]
    d["..sample"] = d[".sample_id"]
    d["..all"] = 1

    if plot_type == 1:
        frame, fill = _type1_frame(d, columns), "ref"
        colours = _palette(palette, _HUE_COLOURS)
        scale = dict(zip(_REF_ORDER, colours, strict=True))
    else:
        frame, fill = _type2_frame(d, columns), "..type"
        colours = _palette(palette, (grey(0.85), "green", "lightyellow", "pink"))
        # R names the scale by status, taking the palette from the back.
        scale = dict(zip(_STATUS_ORDER, colours[::-1], strict=True))

    plot = TubePlot(data=frame, x="value", y="ref")
    plot.layers = [
        LayerSpec(
            geom="GeomCol",
            data=frame,
            mapping={"x": "value", "y": "ref", "fill": fill},
            params={"position": "stack" if plot_type == 2 else "dodge"},
        ),
        LayerSpec(
            geom="GeomVline",
            data=pd.DataFrame({"xintercept": [100.0]}),
            mapping={"xintercept": "xintercept"},
            params={"linetype": "dashed"},
        ),
    ]
    plot.facet, plot.facet_vars, plot.facet_type = "FacetWrap", ["variable"], "wrap"
    plot.labels = {"x": "", "y": "", "subtitle": ""}
    plot.fill_palette = list(scale.values())
    return plot


def _palette(palette, default) -> list[str]:
    """R's ``rep(palette, 4)[1:4]`` -- recycle a short palette to four."""
    if palette is None:
        return list(default)
    values = [palette] if isinstance(palette, str) else list(palette)
    return (values * 4)[:4]


def _stacked(d: pd.DataFrame, columns: list[str], stat) -> pd.DataFrame:
    """One ``calcTubeStat`` per level, stacked with a ``ref`` column."""
    parts = []
    for key, label in _LEVELS:
        part = calc_tube_stat(d, columns, by=key, stat=stat)
        part = part[[c for c in part.columns if c != key]].copy()
        part["ref"] = label
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _melt(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    """``data.table::melt(id.vars = "ref")`` -- ``ref`` varies fastest.

    The measure columns keep their order and become an ordered ``variable``
    factor, which is what the facet strips are laid out by.
    """
    measures = [c for c in frame.columns if c != "ref"]
    out = pd.DataFrame(
        {
            "ref": np.tile(frame["ref"].to_numpy(), len(measures)),
            "variable": np.repeat(measures, rows),
            "value": np.concatenate([frame[c].to_numpy(object) for c in measures]),
        }
    )
    out["variable"] = r_factor(out["variable"], levels=measures)
    return out


def _type1_frame(d: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One bar per level: the percentage of rows where the column is single-valued."""
    stacked = _stacked(d, columns, _count_if_single)

    # data.table's `by = "ref"` keeps groups in order of first appearance.
    order = [label for _, label in _LEVELS]
    summed = (
        stacked.groupby("ref", sort=False, observed=True)
        .sum(numeric_only=True)
        .reindex(order)
        .reset_index()
    )

    out = _melt(summed, len(summed))
    out["value"] = pd.to_numeric(out["value"]) / len(d) * 100
    out["ref"] = r_factor(out["ref"], levels=list(reversed(_REF_ORDER)))
    return out


def _type2_frame(d: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One stacked segment per *group*, coloured by status.

    Deliberately not aggregated: R keeps every group so a column that is clean
    at most locations and broken at a few shows both, rather than an average.
    """
    stacked = _stacked(d, columns, _count_and_status)

    counts = stacked[[f"{c}.count" for c in columns] + ["ref"]]
    counts.columns = [*columns, "ref"]
    types = stacked[[f"{c}.type" for c in columns] + ["ref"]]
    types.columns = [*columns, "ref"]

    out = _melt(counts, len(stacked))
    out["value"] = pd.to_numeric(out["value"]) / len(d) * 100
    out["..type"] = r_factor(_melt(types, len(stacked))["value"], levels=list(_STATUS_ORDER))
    out["ref"] = r_factor(out["ref"], levels=list(reversed(_REF_ORDER)))
    return out


def _count_if_single(values) -> int:
    """R's ``loc.fun`` for plot.type 1: the row count, but only if single-valued."""
    series = pd.Series(values)
    return int(series.notna().sum()) if series.nunique(dropna=False) == 1 else 0


def _count_and_status(values) -> dict:
    """R's ``loc.fun`` for plot.type 2: count plus a traffic-light status."""
    series = pd.Series(values)
    if series.nunique(dropna=False) == 1:
        status = "grey" if series.isna().all() else "green"
    else:
        status = "amber" if series.dropna().nunique() == 1 else "red"
    return {"count": len(series), "type": status}

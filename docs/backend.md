# Using dteval from a web backend

`dteval` is built to run server-side and hand results to a frontend. Two
things follow from that: the core install is deliberately small, and figures
come out as **data** rather than images.

## Install

```bash
pip install dteval                 # numpy, pandas, scipy. Nothing else.
```

Everything beyond the core analysis is an extra, so a slim container stays
slim:

| extra | pulls in | needed for |
|---|---|---|
| `dteval[geo]` | `shapely` | `tube_in_xy_polygon`, `datasets.caz_brd()` |
| `dteval[aqeval]` | `aqeval` | `tube_summary_lat_lon`, `deseason_tube_data(method=2)` |
| `dteval[plots]` | `plotnine` | server-side rendering only — see below |
| `dteval[all]` | all three | |

Calling something without its extra raises an `ImportError` naming the extra to
install, rather than a bare "No module named …". CI has a dedicated job that
runs a full analysis on the **core install only**, so that promise cannot rot.

## Figures without a rendering stack

`tube_plot` and friends return a `TubePlot` — a *description* of the figure,
not an image. `to_spec()` turns it into plain JSON:

```python
import dteval as dte

d = dte.datasets.dt_brd()
plot = dte.tube_plot(d, ".date", ".value", col=".year", ylab="NO2 ug/m3")
spec = plot.to_spec()          # or plot.to_json()
```

```jsonc
{
  "labels": {"x": ".date", "y": "NO<sub>2</sub> &mu;g.m<sup>-3</sup>", "subtitle": ""},
  "facet": null,
  "palette": ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"],
  "theme": "theme_bw",
  "layers": [
    {
      "geom": "GeomPoint",
      "mapping": {"x": ".date", "y": ".value", "colour": ".year"},
      "params": {"na.rm": "TRUE"},
      "data": {"columns": ["site", ".date", ".value", "..."], "nrow": 11273,
               "data": {"...": []}}
    }
  ]
}
```

That is enough to draw the figure with D3, Observable Plot, Chart.js or
similar: each layer says which geom to use, which columns map to which
aesthetics, and carries its own data.

- `to_spec(orient="columns")` (default) is columnar and compact.
- `to_spec(orient="records")` gives row objects, if your frontend prefers them.
- Missing values become `null`, timestamps ISO-8601 strings, categories their
  labels — the result passes through `json.dumps` unchanged.

**The palette is worth using.** It is ggplot2's discrete hue wheel, reproduced
from R's `grDevices::hcl` and verified exact, so a client-drawn chart matches
the colours of the R original.

**Labels contain HTML.** `dte_quickText` renders units as markup —
`NO<sub>2</sub>`, `&mu;g.m<sup>-3</sup>`. Convenient in a browser; strip the
tags if you are rendering to plain text.

### If you do want server-side images

```bash
pip install 'dteval[plots]'
```

```python
plot.draw()                        # a plotnine ggplot object
plot.save("figure.png", width=7, height=4.5, dpi=110)
```

plotnine implements the same grammar as ggplot2, so `theme_bw`, faceting and
the hue palette carry across. Note it cannot render the HTML in the labels, so
`draw()` unwraps them to plain text (`NO2`, `μg.m^-3`).

## Shapes to expect

| function | returns |
|---|---|
| `tag_tube`, `calc_tube_stat`, `deseason_tube_data`, `cluster_tube_data`, `fit_tube_model`, `tube_in_xy_polygon`, `tube_annual_cover` | `DataFrame` |
| `tube_summary`, `tube_summary_sample` | one-row / small `DataFrame` |
| `test_tube_precision` | `{"data": DataFrame, "plot": TubePlot, "lookup": DataFrame, "report": str}` |

`report` is a preformatted human-readable string, reproduced from R
character-for-character — fine to show verbatim in a UI.

Column names keep R's convention (`.value`, `.start_date`, `.sample_id`,
`.annual.pc`, `low.pc`). They are part of the data contract, so they are *not*
renamed to snake_case; only the function names are.

## Serialising DataFrames

`.to_dict(orient="records")` is usually enough, but note pandas emits `NaN`,
`NaT` and `Timestamp` objects, none of which are valid JSON. The helper the
plot spec uses handles all three:

```python
from dteval.plots.tube_plot import _frame_to_json   # columns | records
```

## One thing to know about `.sample_id`

`.sample_id` numbers replicate sets — tubes sharing a location and sampling
period. The numbering is **not** the same as R's (see `docs/parity.md`); the
grouping is. Treat it as an opaque group key, not a stable identifier to store
or compare against R output.

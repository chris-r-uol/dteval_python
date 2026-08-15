"""Diffusion tube plots -- port of ``tubePlot`` / ``tubeTimePlot`` (``R/tube.plots.R``).

``tube_plot`` is the shell every DTEval figure is built on: ``testTubePrecision``
and ``testTubeAccuracy`` both call it and then add their own layers.

The result is a :class:`TubePlot` -- a resolved description of the figure
(layers, mappings, fixed parameters, facets, labels, palette) rather than a
rendered image. That is deliberate: it is the object the parity suite compares
against R, and it can be rendered on demand with :meth:`TubePlot.draw` via
plotnine, which implements the same grammar so the ggplot2 conventions
(``theme_bw``, faceting, the discrete hue palette) carry over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.handlers import DTEvalError, check_tube_data, get_tube_x
from dteval.plots.ggshell import LayerSpec, add_geom, resolve_facet, test_args, tidy_args
from dteval.plots.quicktext import quick_text
from dteval.rcompat.coerce import as_matrix_paste
from dteval.rcompat.palette import hcl_hue_palette
from dteval.rcompat.stats import r_quantile
from dteval.tagging import tag_tube_required

__all__ = ["TubePlot", "tube_plot", "tube_time_plot"]

#: plot.type values understood by tubePlot, in R's order (tube.plots.R:477).
PLOT_TYPES = (
    "point", "line", "box", "band", "surface", "smooth", "polygon", "none", "ggsmooth",
)

_NUMERIC_GRADIENT = ("#132B43", "#56B1F7")  # ggplot2's default continuous scale


@dataclass
class TubePlot:
    """A resolved DTEval figure."""

    data: pd.DataFrame
    x: str
    y: str
    layers: list[LayerSpec] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    facet: str = "FacetNull"
    facet_vars: list[str] = field(default_factory=list)
    facet_type: str = "wrap"
    palette: list[str] | None = None
    fill_palette: list[str] | None = None
    theme: str = "theme_bw"

    def parity_spec(self) -> dict[str, Any]:
        """Internal form used by the parity comparator (data stays a DataFrame)."""
        return {
            "labels": dict(self.labels),
            "facet": self.facet,
            "facet_vars": list(self.facet_vars),
            "layers": [layer.to_spec() for layer in self.layers],
        }

    def to_spec(self, orient: str = "columns") -> dict[str, Any]:
        """A JSON-serialisable description of the figure.

        Everything a client needs to draw the plot itself: per layer the geom,
        the data, which columns map to which aesthetics, and any constant
        aesthetics; plus the axis labels, facet variables and palette.

        This is the intended output for a web frontend -- the backend need not
        render anything, and ``dteval[plots]`` need not be installed. ``orient``
        is ``"columns"`` (default, compact) or ``"records"`` (row objects).

        NaN becomes ``None``, timestamps become ISO-8601 strings and categories
        become their labels, so the result survives ``json.dumps`` unchanged.
        """
        return {
            "labels": {k: v for k, v in self.labels.items() if v is not None},
            "facet": {"type": self.facet_type, "vars": list(self.facet_vars)}
            if self.facet_vars
            else None,
            "palette": list(self.palette) if self.palette else None,
            "fill_palette": list(self.fill_palette) if self.fill_palette else None,
            "theme": self.theme,
            "layers": [
                {
                    "geom": layer.geom,
                    "mapping": dict(layer.mapping),
                    "params": {k: _jsonable(v) for k, v in layer.params.items()},
                    "data": _frame_to_json(layer.data, orient),
                }
                for layer in self.layers
            ],
        }

    def to_json(self, orient: str = "columns", **kwargs) -> str:
        """:meth:`to_spec` serialised with ``json.dumps``."""
        import json

        return json.dumps(self.to_spec(orient=orient), **kwargs)

    # -- rendering ---------------------------------------------------------
    def draw(self):
        """Render with plotnine and return the ``ggplot`` object."""
        return _to_plotnine(self)

    def save(self, path: str, **kwargs):
        return self.draw().save(path, **kwargs)

    def __repr__(self) -> str:
        geoms = ", ".join(layer.geom for layer in self.layers) or "no layers"
        return f"<TubePlot {self.x} vs {self.y}: {geoms}>"


def tube_plot(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    plot_type: str | list[str] | None = None,
    **kwargs,
) -> TubePlot:
    """Port of ``tubePlot``.

    ``plot_type`` may be omitted, in which case it is inferred from any
    ``<type>.<arg>`` keywords present and defaults to ``"point"``.
    """
    xargs = tidy_args(dict(kwargs))
    xargs = {**{"grid.borders": 0.05, "title": "", "auto.text": True}, **xargs}

    d2 = data.data if isinstance(data, TubePlot) else data
    base: TubePlot | None = data if isinstance(data, TubePlot) else None
    if "new.data" in xargs:
        d2 = xargs["new.data"]

    # Tag anything referenced by name that is not present yet.
    required = [v for v in xargs.values() if isinstance(v, str)]
    d2 = tag_tube_required(d2, required=[v for v in (x, y) if v] + required)

    classes = test_args(xargs, d2)
    to_check = [xargs[k] for k, v in classes.items() if v == "data"]
    if to_check:
        d2 = check_tube_data(d2, to_check)

    if x is None and y is None:
        raise DTEvalError("[plotTube]> Sorry, need at least one of: x,y \n")
    if y is None:
        d2 = d2.copy()
        d2[".index"] = np.arange(1, len(d2) + 1)
        y = ".index"
    elif x is None:
        d2 = d2.copy()
        d2[".index"] = np.arange(1, len(d2) + 1)
        x = ".index"

    d2 = d2.copy()
    d2[x] = get_tube_x(d2, x, if_err="stop<<tubePlot>>x")
    d2[y] = get_tube_x(d2, y, if_err="stop<<tubePlot>>x")

    if "group" in xargs:
        d2 = check_tube_data(d2, xargs["group"], n_x=1, if_err="stop<<tubePlot>>group")
    if "facet" in xargs:
        facets = xargs["facet"]
        facets = [facets] if isinstance(facets, str) else list(facets)
        d2 = check_tube_data(d2, facets, n_x=2, if_err="stop<<tubePlot>>facet")

    xargs.setdefault("xlab", x)
    xargs.setdefault("ylab", y)
    xargs.setdefault("title", "")

    types = _resolve_plot_types(plot_type, xargs)

    plot = base or TubePlot(data=d2, x=x, y=y)
    plot.data = d2
    plot.x, plot.y = x, y

    for kind in types:
        layer = _build_layer(kind, d2, x, y, xargs)
        if layer is not None:
            plot.layers.extend(layer)

    plot.facet, plot.facet_vars = resolve_facet(xargs)
    plot.facet_type = xargs.get("facet.type", "wrap")
    plot.palette, plot.fill_palette = _resolve_palettes(xargs, d2, classes)

    auto = bool(xargs.get("auto.text", True))
    plot.labels = {
        "x": quick_text(xargs.get("xlab"), auto),
        "y": quick_text(xargs.get("ylab"), auto),
        "subtitle": quick_text(xargs.get("title"), auto),
    }
    return plot


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _frame_to_json(frame: pd.DataFrame, orient: str) -> Any:
    """Serialise a layer's data, preserving types a frontend can use.

    Timestamps become ISO-8601 strings, categoricals their labels, and every
    flavour of missing value becomes ``null`` -- pandas alone would emit ``NaN``
    or ``NaT``, which are not valid JSON.
    """
    out: dict[str, list[Any]] = {}
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            values = [None if pd.isna(v) else v.isoformat() for v in series]
        elif isinstance(series.dtype, pd.CategoricalDtype):
            values = [None if pd.isna(v) else str(v) for v in series]
        elif pd.api.types.is_bool_dtype(series.dtype):
            values = [None if pd.isna(v) else bool(v) for v in series]
        elif pd.api.types.is_integer_dtype(series.dtype):
            values = [None if pd.isna(v) else int(v) for v in series]
        elif pd.api.types.is_float_dtype(series.dtype):
            values = [None if pd.isna(v) else float(v) for v in series]
        else:
            values = [None if pd.isna(v) else str(v) for v in series]
        out[str(name)] = values

    if orient == "columns":
        return {"columns": list(out), "nrow": len(frame), "data": out}
    if orient == "records":
        names = list(out)
        return [dict(zip(names, row, strict=True)) for row in zip(*out.values(), strict=True)]
    raise ValueError(f"orient must be 'columns' or 'records', got {orient!r}")


def _resolve_plot_types(plot_type, xargs: dict[str, Any]) -> list[str]:
    """Port of the ``plot.type`` sniffing block (``tube.plots.R:477``).

    Any ``<type>.<arg>`` keyword implies that layer type, so
    ``tubePlot(d, x, y, smooth.col = "red")`` draws a smooth without being
    told to.
    """
    candidates: list[str] = []
    if plot_type is not None:
        candidates += [plot_type] if isinstance(plot_type, str) else list(plot_type)
    candidates += list(xargs.keys())

    resolved: list[str] = []
    for token in candidates:
        for kind in PLOT_TYPES:
            if token == kind or str(token).startswith(f"{kind}."):
                if kind not in resolved:
                    resolved.append(kind)
    return resolved or ["point"]


def _build_layer(kind: str, d2: pd.DataFrame, x: str, y: str, xargs: dict) -> list[LayerSpec] | None:
    args = tidy_args(xargs, kind)
    if args.get("..test") != "OK":
        return None

    if kind == "none":
        return None

    if kind == "point":
        return _layer_point(args, d2, x, y)
    if kind == "line":
        return _layer_line(args, d2, x, y)
    if kind == "box":
        return _layer_box(args, d2, x, y)
    if kind == "band":
        return _layer_band(args, d2, x, y)
    if kind == "polygon":
        return _layer_polygon(args, d2)
    if kind == "ggsmooth":
        return [
            add_geom(
                {**args, "x": x, "y": y},
                d2,
                "GeomSmooth",
                defaults={"na.rm": True},
            )
        ]
    if kind in ("smooth", "surface"):
        raise NotImplementedError(
            f"plot.type={kind!r} needs fitTubeModel, which is not ported yet "
            "(tracked as task 8: fit/deseason/cluster)"
        )
    return None


def _layer_point(args: dict, d2: pd.DataFrame, x: str, y: str) -> list[LayerSpec]:
    data = d2
    if "overplot" in args:
        data, args = _apply_overplot(args, d2, x, y)
    return [add_geom({**args, "x": x, "y": y}, data, "GeomPoint", defaults={"na.rm": True})]


def _apply_overplot(args: dict, d2: pd.DataFrame, x: str, y: str):
    """Port of the ``overplot`` option: collapse coincident points.

    ``overplot="mean"`` averages the numeric aesthetics over each x/y/group
    combination; ``overplot="count"`` counts them and, if nothing else is
    mapped to size, maps the count to point size.
    """
    classes = test_args(args, d2)
    by = [x, y]
    for key in ("facet", "group"):
        if key in args:
            v = args[key]
            by += [v] if isinstance(v, str) else list(v)
    numeric: list[str] = []
    for name, value in args.items():
        if classes.get(name) != "data":
            continue
        series = get_tube_x(d2, value)
        if series is not None and pd.api.types.is_numeric_dtype(pd.Series(series).dtype):
            numeric.append(value)
        else:
            by.append(value)

    by = list(dict.fromkeys(by))
    numeric = [v for v in dict.fromkeys(numeric) if v not in by]

    data = d2
    if not numeric:
        data = d2.copy()
        data["..dummy"] = 1
        numeric = ["..dummy"]

    mode = args["overplot"]
    args = dict(args)
    if mode == "mean":
        from dteval.rcompat.stats import r_mean

        data = calc_tube_stat(data, numeric, by=by, stat=lambda v: {"mean": r_mean(v, na_rm=True)})
        suffix = ".mean"
    elif mode == "count":
        data = calc_tube_stat(
            data, numeric, by=by, stat=lambda v: {"count": int(pd.Series(v).notna().sum())}
        )
        suffix = ".count"
        if "..dummy.count" in data.columns and "size" not in args:
            args["size"] = "..dummy.count"
    else:
        return d2, args  # R ignores an unrecognised overplot

    for name, value in list(args.items()):
        if value in numeric:
            args[name] = f"{value}{suffix}"
    return data, args


def _layer_line(args: dict, d2: pd.DataFrame, x: str, y: str) -> list[LayerSpec]:
    ordered = d2.sort_values(x, kind="stable")
    return [add_geom({**args, "x": x, "y": y}, ordered, "GeomPath", defaults={"na.rm": True})]


def _layer_box(args: dict, d2: pd.DataFrame, x: str, y: str) -> list[LayerSpec]:
    """Port of the ``box`` type (``tube.plots.R:911``).

    Without an explicit ``group``, R builds one by pasting the discrete axis
    together with any other data-mapped aesthetics, so that e.g. a fill by site
    splits the boxes rather than merging them.
    """
    data = d2
    args = dict(args)
    if "group" not in args:
        classes = test_args(args, d2)
        extra = [args[k] for k, v in classes.items() if v == "data"]
        extra = [e for e in dict.fromkeys(extra) if e not in (x, y)]
        # "this'll die if both are numeric" -- the R comment; the discrete axis
        # is whichever of x/y is not numeric.
        axis = y if pd.api.types.is_numeric_dtype(d2[x].dtype) else x
        data = d2.copy()
        data["..group"] = as_matrix_paste(data[[axis, *extra]], sep="&")
        args["group"] = "..group"

    return [
        add_geom(
            {**args, "x": x, "y": y},
            data,
            "GeomBoxplot",
            defaults={"na.rm": True, "fill": "grey"},
        )
    ]


def _layer_polygon(args: dict, d2: pd.DataFrame) -> list[LayerSpec]:
    poly = args.get("polygon")
    coords = _polygon_coords(poly)
    return [
        add_geom(
            {**args, "x": "X", "y": "Y"},
            coords,
            "GeomPolygon",
            defaults={"na.rm": True, "colour": "blue", "fill": "blue", "alpha": 0.25},
        )
    ]


def _polygon_coords(poly) -> pd.DataFrame:
    """``sf::st_coordinates`` equivalent for the polygon sources DTEval accepts."""
    from dteval.datasets import PolygonSource

    if isinstance(poly, pd.DataFrame):
        return poly
    if isinstance(poly, PolygonSource):
        rows = []
        for i, geom in enumerate(poly.geoms, start=1):
            for xx, yy in _exterior_coords(geom):
                rows.append({"X": xx, "Y": yy, "L1": i})
        return pd.DataFrame(rows)
    raise DTEvalError("[tubePlot]> polygon source not understood")


def _exterior_coords(geom):
    if hasattr(geom, "geoms"):
        for sub in geom.geoms:
            yield from _exterior_coords(sub)
        return
    if hasattr(geom, "exterior"):
        yield from list(geom.exterior.coords)
        return
    yield from list(geom.coords)


def _layer_band(args: dict, d2: pd.DataFrame, x: str, y: str) -> list[LayerSpec]:
    """Port of the ``band`` type: a quantile ribbon plus its centre line."""
    probs = args.get("band.probs", args.get("probs", [0, 0.5, 1]))
    direction = args.get("cheat", "by-x")

    classes = test_args(args, d2)
    by = [x, y]
    for key in ("facet", "group"):
        if key in args:
            v = args[key]
            by += [v] if isinstance(v, str) else list(v)
    numeric: list[str] = []
    for name, value in args.items():
        if classes.get(name) != "data":
            continue
        series = get_tube_x(d2, value)
        if series is not None and pd.api.types.is_numeric_dtype(pd.Series(series).dtype):
            numeric.append(value)
        else:
            by.append(value)
    by = list(dict.fromkeys(by))
    numeric = [v for v in dict.fromkeys(numeric) if v not in by]

    data = d2
    if not numeric:
        data = d2.copy()
        data["..dummy"] = 1
        numeric = ["..dummy"]

    target, other = (y, x) if direction == "by-x" else (x, y)
    prefix = "y" if direction == "by-x" else "x"

    def stat(v):
        q = r_quantile(v, [probs[1], probs[0], probs[2]], na_rm=True)
        return {f"{prefix}.mid": q[0], f"{prefix}.low": q[1], f"{prefix}.hi": q[2]}

    group_by = [c for c in dict.fromkeys([*numeric, *by]) if c != target]
    stats = calc_tube_stat(data, target, by=group_by, stat=stat)
    stats = stats.sort_values(other, kind="stable")

    args = dict(args)
    if "colour" in args and "group" not in args and classes.get("colour") == "data":
        args["group"] = args["colour"]
    if "fill" in args and "group" not in args and classes.get("fill") == "data":
        args["group"] = args["fill"]

    mid, low, hi = (
        f"{target}.{prefix}.mid",
        f"{target}.{prefix}.low",
        f"{target}.{prefix}.hi",
    )
    if direction == "by-x":
        ribbon_args = {**args, "x": x, "y": mid, "ymin": low, "ymax": hi}
    else:
        ribbon_args = {**args, "y": y, "x": mid, "xmin": low, "xmax": hi}

    ribbon = add_geom(
        ribbon_args, stats, "GeomRibbon",
        defaults={"na.rm": True, "fill": "grey", "alpha": 0.5},
        drops={"colour"},
    )
    line = add_geom(
        ribbon_args, stats, "GeomLine", defaults={"na.rm": True}, drops={"alpha"}
    )
    return [ribbon, line]


def _resolve_palettes(xargs: dict, d2: pd.DataFrame, classes: dict[str, str]):
    """Port of the palette block (``tube.plots.R:1181``).

    A colour aesthetic mapped to data gets ggplot2's default scale: the blue
    gradient for a numeric column, or the evenly spaced HCL hue wheel for a
    discrete one.
    """
    def default_for(keys: tuple[str, ...]):
        mapped = [k for k, v in classes.items() if v == "data" and (k in keys or any(k.endswith(f".{t}") for t in keys))]
        if not mapped:
            return None
        values = get_tube_x(d2, xargs[mapped[0]])
        if values is None:
            return None
        series = pd.Series(values)
        if pd.api.types.is_numeric_dtype(series.dtype):
            return list(_NUMERIC_GRADIENT)
        return hcl_hue_palette(series.nunique(dropna=True))

    palette = xargs.get("palette") or default_for(("colour", "col", "color"))
    fill_palette = xargs.get("fill.palette") or xargs.get("palette") or default_for(("fill",))
    return palette, fill_palette


def tube_time_plot(data, x: str | None = None, y: str | None = None, **kwargs) -> TubePlot:
    """Port of ``tubeTimePlot`` -- ``tubePlot`` with ``.date`` on the free axis."""
    if x is None and y is not None:
        x = ".date"
    if y is None and x is not None:
        y = ".date"
    return tube_plot(data, x=x, y=y, **kwargs)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_GEOM_TO_PLOTNINE = {
    "GeomPoint": "geom_point",
    "GeomPath": "geom_path",
    "GeomLine": "geom_line",
    "GeomSmooth": "geom_smooth",
    "GeomRibbon": "geom_ribbon",
    "GeomBoxplot": "geom_boxplot",
    "GeomPolygon": "geom_polygon",
    "GeomRaster": "geom_raster",
    "GeomContour": "geom_contour",
    "GeomCol": "geom_col",
    "GeomVline": "geom_vline",
}

# plotnine spells a few ggplot2 aesthetics differently.
_AES_RENAME = {"colour": "color"}


def _to_plotnine(plot: TubePlot):
    try:
        import plotnine as p9
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Rendering needs plotnine. Install it with: pip install "
            "'dteval[plots]' -- or use TubePlot.to_spec() and draw client-side "
            "(see docs/backend.md)."
        ) from exc

    # tubePlot's base is theme_bw() with transparent facet strips
    # (tube.plots.R:471) -- hence white strip headers rather than grey ones.
    p = p9.ggplot(plot.data) + p9.theme_bw()
    # matplotlib has no "transparent" colour keyword; "none" is its equivalent.
    p = p + p9.theme(strip_background=p9.element_rect(fill="none"))

    for layer in plot.layers:
        fn = getattr(p9, _GEOM_TO_PLOTNINE[layer.geom])
        mapping = {_AES_RENAME.get(k, k): v for k, v in layer.mapping.items()}
        params = {
            _AES_RENAME.get(k, k): v
            for k, v in layer.params.items()
            if k not in ("na.rm", "group")
        }
        p = p + fn(p9.aes(**mapping), data=layer.data, **params)

    p = _add_date_scales(p, plot)
    p = _add_palettes(p, plot)

    if plot.facet_vars:
        if plot.facet == "FacetWrap":
            p = p + p9.facet_wrap(plot.facet_vars)
        elif plot.facet_type == "grid.col":
            p = p + p9.facet_grid(cols=plot.facet_vars[:1])
        else:
            p = p + p9.facet_grid(rows=plot.facet_vars[:1])

    p = p + p9.labs(
        x=_strip_markup(plot.labels.get("x")),
        y=_strip_markup(plot.labels.get("y")),
        subtitle=_strip_markup(plot.labels.get("subtitle")),
    )
    return p


def _add_date_scales(p, plot: TubePlot):
    """Label date axes the way ggplot2 does.

    ggplot2 picks a date format from the break spacing, so a multi-year axis
    reads ``2022 2023 2024`` rather than ``2022-01-01 ...``. plotnine defaults
    to the full date, which looks wrong next to the R original. This
    approximates ggplot2's choice from the data range -- it is a rendering
    convention, not a value, so it is not part of the parity contract.
    """
    import plotnine as p9

    for axis, column in (("x", plot.x), ("y", plot.y)):
        if column not in plot.data.columns:
            continue
        series = plot.data[column]
        if not pd.api.types.is_datetime64_any_dtype(series.dtype):
            continue
        span_days = (series.max() - series.min()).days if series.notna().any() else 0
        if span_days > 730:
            fmt = "%Y"
        elif span_days > 60:
            fmt = "%b %Y"
        else:
            fmt = "%b %d"
        scale = p9.scale_x_date if axis == "x" else p9.scale_y_date
        p = p + scale(date_labels=fmt)
    return p


def _add_palettes(p, plot: TubePlot):
    """Apply the colour/fill palettes tubePlot resolved.

    For a discrete mapping these are ggplot2's evenly spaced HCL hues, computed
    in :mod:`dteval.rcompat.palette` and verified against ``grDevices::hcl``.
    """
    import plotnine as p9

    mapped_colour = any("colour" in layer.mapping for layer in plot.layers)
    mapped_fill = any("fill" in layer.mapping for layer in plot.layers)

    if plot.palette and mapped_colour:
        if len(plot.palette) == 2 and plot.palette[0].startswith("#13"):
            p = p + p9.scale_color_gradientn(colors=list(plot.palette))
        else:
            p = p + p9.scale_color_manual(values=list(plot.palette))
    if plot.fill_palette and mapped_fill:
        if len(plot.fill_palette) == 2 and plot.fill_palette[0].startswith("#13"):
            p = p + p9.scale_fill_gradientn(colors=list(plot.fill_palette))
        else:
            p = p + p9.scale_fill_manual(values=list(plot.fill_palette))
    return p


def _strip_markup(text: str | None) -> str | None:
    """Render DTEval's markdown-ish labels as plain text.

    ``dte_quickText`` emits HTML for ``ggtext::element_markdown``. plotnine has
    no markdown renderer, so subscripts and the micro entity are unwrapped to
    their nearest plain-text form rather than shown as literal tags.
    """
    if text is None:
        return None
    import re

    out = text.replace("&mu;", "μ").replace("<br>", "\n")
    out = re.sub(r"<sub>(.*?)</sub>", r"\1", out)
    out = re.sub(r"<sup>(.*?)</sup>", r"^\1", out)
    return re.sub(r"</?[a-z]+>", "", out)

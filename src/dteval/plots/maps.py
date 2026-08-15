"""Diffusion tube maps -- port of ``R/tube.maps.R``.

Two maps, matching the two the R package offers:

* :func:`tube_map` -- a static figure. R draws it as a ``ggplot`` with an
  ``OpenStreetMap`` ESRI raster annotated underneath the tube layers.
* :func:`leaflet_tube_map` -- an interactive map, built in R with ``leaflet``.

Neither renders here. Both return a *description* -- a :class:`~dteval.plots.tube_plot.TubePlot`
carrying a basemap request, and a :class:`LeafletMap` carrying an ordered list
of leaflet calls -- which is what a web frontend actually wants, and what the
parity suite compares. :meth:`LeafletMap.draw` will render with folium if
``dteval[plots]`` is installed, and :meth:`TubePlot.draw` already renders with
plotnine, but the basemap tiles themselves are always the client's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dteval.handlers import get_tube_x
from dteval.plots.ggshell import LayerSpec, tidy_args
from dteval.plots.tube_plot import TubePlot, _frame_to_json, _jsonable, tube_plot
from dteval.tagging import tag_tube, tag_tube_required

__all__ = ["LeafletMap", "leaflet_tube_map", "tube_map"]

#: ``names(formals(leaflet::addCircleMarkers))`` -- the filter R applies via
#: ``dte_localArgsTidies(tidy.ref=)``.
_CIRCLE_MARKER_ARGS = frozenset(
    """map lng lat radius layerId group stroke color weight opacity fill fillColor
    fillOpacity dashArray popup popupOptions label labelOptions options
    clusterOptions clusterId data""".split()
)

#: leafletTubeMap's subset of tubePlot's plot.type values (tube.maps.R:361).
_LEAFLET_PLOT_TYPES = ("point", "surface", "polygon", "none")


def tube_map(data, x: str | None = None, y: str | None = None, **kwargs) -> TubePlot:
    """Port of ``tubeMap`` -- tube locations on a static basemap.

    ``x`` and ``y`` default to ``.longitude`` and ``.latitude``. The map extent
    is the data range grown by ``grid.borders`` (0.05, i.e. 5% each way) and is
    attached to the returned plot as ``limits``, together with a ``basemap``
    request naming the tile provider R asks ``OpenStreetMap`` for.

    The raster itself is not fetched: R's basemap comes from a Java tile
    stack, and re-fetching different tiles would not make the figure more
    faithful. Everything else -- the layers, the extent, the zeroed scale
    expansion and the blanked axes -- is reproduced, and gated against R.
    """
    xargs = dict(kwargs)
    for alias in ("col", "color"):
        if alias in xargs:
            xargs["colour"] = xargs.pop(alias)

    x = ".longitude" if x is None else x
    y = ".latitude" if y is None else y

    facet = xargs.get("facet")
    required = [x, y] + ([facet] if isinstance(facet, str) else list(facet or []))
    d2 = tag_tube_required(data, required=required)

    xs = get_tube_x(d2, x, if_err="stop<<tubeMap>>x")
    ys = get_tube_x(d2, y, if_err="stop<<tubeMap>>y")
    borders = xargs.setdefault("grid.borders", 0.05)

    limits = {
        x: list(_expand(np.nanmin(xs), np.nanmax(xs), borders)),
        y: list(_expand(np.nanmin(ys), np.nanmax(ys), borders)),
    }

    # R calls tubePlot twice -- once to build the figure the raster is annotated
    # onto, then again with that figure as `data` to add the coord and theme.
    # tubePlot appends layers to a plot it is handed, so the tube layers land
    # twice. That is visible in the R output, so it is reproduced here.
    plot = tube_plot(d2, x=x, y=y, **xargs)
    # ggplot2's expand_limits() is a geom_blank layer over the two corners, and
    # it lands between the two sets of tube layers.
    plot.layers.append(
        LayerSpec(
            geom="GeomBlank",
            data=pd.DataFrame({"x": limits[x], "y": limits[y]}),
            mapping={"x": "x", "y": "y"},
            params={},
        )
    )
    plot = tube_plot(plot, x=x, y=y, **xargs)

    plot.limits = limits
    plot.basemap = {"provider": "esri", "projection": "WGS84", "bbox": limits}
    plot.coord = "coord_quickmap"
    plot.expand = {"x": [0, 0], "y": [0, 0]}
    plot.blank_axes = True
    return plot


def _expand(low: float, high: float, fraction: float) -> tuple[float, float]:
    """R's local ``exp()``: grow a range by a fraction of its own width."""
    pad = (high - low) * fraction
    return low - pad, high + pad


@dataclass
class LeafletMap:
    """A resolved interactive map: an ordered list of leaflet calls.

    Mirrors the structure of a leaflet htmlwidget (``x$calls``), so the parity
    suite can compare marker coordinates, colours, polygon rings and legend
    breaks against R directly.
    """

    data: pd.DataFrame
    calls: list[dict[str, Any]] = field(default_factory=list)

    def parity_spec(self) -> dict[str, Any]:
        return {"calls": [{"method": c["method"], "args": c["args"]} for c in self.calls]}

    def to_spec(self, orient: str = "columns") -> dict[str, Any]:
        """A JSON-serialisable description, for a leaflet/maplibre frontend."""
        out = []
        for call in self.calls:
            args = {}
            for name, value in call["args"].items():
                if isinstance(value, pd.DataFrame):
                    args[name] = _frame_to_json(value, orient)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    args[name] = [_jsonable(v) for v in value]
                else:
                    args[name] = _jsonable(value)
            out.append({"method": call["method"], "args": args})
        return {"calls": out}

    def to_json(self, orient: str = "columns", **kwargs) -> str:
        import json

        return json.dumps(self.to_spec(orient=orient), **kwargs)

    def draw(self):
        """Render with folium and return the ``folium.Map``."""
        return _to_folium(self)

    def save(self, path: str, **kwargs):
        return self.draw().save(path, **kwargs)

    def __repr__(self) -> str:
        methods = ", ".join(call["method"] for call in self.calls) or "empty"
        return f"<LeafletMap: {methods}>"


def leaflet_tube_map(
    data,
    x: str | None = None,
    y: str | None = None,
    plot_type: str | list[str] | None = None,
    **kwargs,
) -> LeafletMap:
    """Port of ``leafletTubeMap`` -- tube locations on an interactive map.

    ``plot_type`` is any of ``point`` (default), ``surface``, ``polygon`` and
    ``none``, and, as in ``tubePlot``, a prefixed argument such as
    ``point.color`` implies its own layer.
    """
    xargs = tidy_args(dict(kwargs))
    xargs = {"grid.borders": 0.05, "auto.text": True, **xargs}

    if isinstance(data, LeafletMap):
        previous, d2 = data, data.data
    else:
        previous, d2 = None, data
    if "new.data" in xargs:
        d2 = xargs["new.data"]
    d2 = tag_tube(d2)

    if previous is None:
        out = LeafletMap(
            data=d2,
            calls=[{"method": "addProviderTiles", "args": {"provider": "CartoDB.Positron"}}],
        )
    else:
        out = LeafletMap(data=d2, calls=list(previous.calls))

    # A prefixed argument names its own layer: `point.color=` implies "point".
    candidates = ([plot_type] if isinstance(plot_type, str) else list(plot_type or []))
    candidates = candidates + list(xargs)
    resolved = []
    for name in candidates:
        for kind in _LEAFLET_PLOT_TYPES:
            if name == kind or name.startswith(f"{kind}."):
                name = kind
                break
        if name in _LEAFLET_PLOT_TYPES and name not in resolved:
            resolved.append(name)
    if not resolved:
        resolved = ["point"]

    for kind in resolved:
        if kind == "polygon" and xargs.get("polygon") is not None:
            out.calls.append(
                {"method": "addPolygons", "args": {"data": _rings(xargs["polygon"])}}
            )
        elif kind == "surface":
            _add_surface(out, d2, xargs)
        elif kind == "point":
            _add_points(out, d2, xargs)
    return out


def _add_points(out: LeafletMap, d2: pd.DataFrame, xargs: dict) -> None:
    """``addCircleMarkers`` at every tube location."""
    args = tidy_args(dict(xargs), "point")
    # leaflet spells it `color`, and takes `radius` where ggplot2 takes `size`.
    for src, dst in (("colour", "color"), ("size", "radius")):
        if src in args:
            args[dst] = args.pop(src)
    if args.get("..test") != "OK":
        return
    args = {k: v for k, v in args.items() if k in _CIRCLE_MARKER_ARGS}

    if "color" in args and get_tube_x(d2, args["color"]) is not None:
        values = pd.to_numeric(get_tube_x(d2, args["color"]), errors="coerce")
        args["color"] = colour_numeric(values, "Spectral")

    # leaflet's own addCircleMarkers defaults, spelled out so the spec is
    # self-contained for a frontend that is not leaflet.
    args = {"lng": list(d2[".longitude"]), "lat": list(d2[".latitude"]), "radius": 2, **args}
    args.setdefault("color", "#03F")
    args.setdefault("fillColor", args["color"])
    out.calls.append({"method": "addCircleMarkers", "args": args})


def _add_surface(out: LeafletMap, d2: pd.DataFrame, xargs: dict) -> None:
    """A fitted concentration surface, as a raster plus labelled contours."""
    from dteval.fit import fit_tube_model

    passthrough = ("too.far", "model", "simplify", "force.positive",
                   "grid.resolution", "grid.borders")
    fit_args = {
        "tube": ".value",
        "inputs": [".longitude", ".latitude"],
        "new_data": "input.ranges",
        "simplify": True,
        **{k.replace(".", "_"): xargs[k] for k in passthrough if k in xargs},
    }
    fitted = fit_tube_model(d2, **fit_args)

    lon = pd.unique(fitted[".longitude"])
    lat = pd.unique(fitted[".latitude"])
    pred = fitted[".value.pred"].to_numpy(float)
    breaks = _pretty(pred)

    out.calls.append(
        {
            "method": "addRasterImage",
            "args": {
                "bounds": [
                    [float(np.nanmin(lat)), float(np.nanmin(lon))],
                    [float(np.nanmax(lat)), float(np.nanmax(lon))],
                ],
                "nrows": len(lon),
                "ncols": len(lat),
                "values": [_jsonable(v) for v in pred],
                "opacity": 0.3,
                "palette": "Spectral",
                "reverse": True,
            },
        }
    )
    out.calls.append(
        {"method": "addLegend", "args": {"palette": "Spectral", "values": list(breaks)}}
    )

    grid = pred.reshape(len(lon), len(lat))
    for level in [n * 5 for n in range(1, 7)]:
        for line in _contour_lines(np.asarray(lon, float), np.asarray(lat, float), grid, level):
            out.calls.append(
                {
                    "method": "addPolylines",
                    "args": {"lng": line["x"], "lat": line["y"], "group": level,
                             "label": level},
                }
            )
            out.calls.append(
                {
                    "method": "addLabelOnlyMarkers",
                    "args": {"lng": line["x"][0], "lat": line["y"][0], "group": level,
                             "label": level},
                }
            )


def colour_numeric(
    values, palette: str = "Spectral", reverse: bool = False, na_colour: str = "#808080"
) -> list[str]:
    """Port of ``leaflet::colorNumeric`` for the ColorBrewer palettes we use.

    Values are rescaled onto [0, 1] across their own range, then looked up on
    the palette. The interpolation happens in **CIE Lab**, not sRGB, because
    that is what ``scales::colour_ramp`` -- which leaflet calls -- does;
    interpolating in sRGB instead is off by a couple of levels per channel.
    """
    anchors = list(_BREWER[palette])
    if reverse:
        anchors.reverse()
    lab = _srgb_to_lab(np.array([[int(a[i : i + 2], 16) for i in (1, 3, 5)] for a in anchors]))

    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    low, high = np.nanmin(v), np.nanmax(v)
    scaled = np.zeros_like(v) if high == low else (v - low) / (high - low)

    position = np.clip(scaled, 0, 1) * (len(anchors) - 1)
    out = []
    for p, raw in zip(position, v, strict=True):
        if np.isnan(raw):
            out.append(na_colour)
            continue
        i = min(int(np.floor(p)), len(anchors) - 2)
        t = p - i
        r, g, b = _lab_to_srgb(lab[i] * (1 - t) + lab[i + 1] * t)
        out.append(f"#{r:02X}{g:02X}{b:02X}")
    return out


#: ColorBrewer anchors, as leaflet ships them.
_BREWER = {
    "Spectral": ["#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#FFFFBF",
                 "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2"],
}

#: sRGB (D65) <-> XYZ, and the D65 white point, as farver uses them.
_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6 / 29


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, float) / 255
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _lab_to_srgb(lab: np.ndarray) -> tuple[int, int, int]:
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4 / 29)) * _WHITE_D65
    linear = _XYZ_TO_RGB @ xyz
    c = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055)
    return tuple(int(v) for v in np.clip(np.rint(c * 255), 0, 255).astype(int))


def _pretty(values) -> list[float]:
    """R's ``pretty()``: round breaks covering the data at a 1/2/5 x 10^k step.

    Matches R for any range with width. A *zero-width* range does not: R's
    ``R_pretty`` takes a separate branch there and invents a span around the
    value (``pretty(c(5, 5))`` is ``0 5``), where this returns the single
    value. It only arises for a legend on a perfectly flat fitted surface, and
    one break is the more useful answer for that; noted in docs/parity.md.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return []
    low, high = float(v.min()), float(v.max())
    if low == high:
        return [low]
    raw = (high - low) / 5
    power = 10 ** np.floor(np.log10(raw))
    step = min((s for s in (1, 2, 5, 10) if s * power >= raw), default=10) * power
    start = np.floor(low / step) * step
    stop = np.ceil(high / step) * step
    n = int(round((stop - start) / step))
    return [float(start + i * step) for i in range(n + 1)]


def _rings(polygon) -> list[list[list[float]]]:
    """Coordinate rings from a shapely geometry, GeoJSON-style [[lon, lat], ...]."""
    try:
        geoms = list(getattr(polygon, "geoms", [polygon]))
    except TypeError:  # pragma: no cover - not a shapely object
        geoms = [polygon]
    rings = []
    for geom in geoms:
        boundary = getattr(geom, "exterior", None) or geom
        rings.append([[float(a), float(b)] for a, b in boundary.coords])
    return rings


def _contour_lines(x, y, z, level: float) -> list[dict[str, list[float]]]:
    """R's ``grDevices::contourLines`` for one level, via matplotlib."""
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "the surface layer needs matplotlib. Install it with: "
            "pip install 'dteval[plots]'"
        ) from exc

    figure = plt.figure()
    try:
        contours = plt.contour(x, y, z.T, levels=[level])
        out = []
        for path in contours.get_paths():
            for poly in path.to_polygons(closed_only=False):
                if len(poly) > 1:
                    out.append({"x": [float(v) for v in poly[:, 0]],
                                "y": [float(v) for v in poly[:, 1]]})
        return out
    finally:
        plt.close(figure)


def _to_folium(spec: LeafletMap):
    """Render a :class:`LeafletMap` with folium."""
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "LeafletMap.draw() needs folium. Install it with: pip install 'dteval[plots]'"
        ) from exc

    lat, lon = spec.data[".latitude"], spec.data[".longitude"]
    fmap = folium.Map(
        location=[float(lat.mean()), float(lon.mean())], tiles="CartoDB positron"
    )
    for call in spec.calls:
        args = call["args"]
        if call["method"] == "addCircleMarkers":
            colours = args.get("color")
            for i, (lg, lt) in enumerate(zip(args["lng"], args["lat"], strict=True)):
                if pd.isna(lg) or pd.isna(lt):
                    continue
                colour = colours[i] if isinstance(colours, list) else colours
                folium.CircleMarker(
                    location=[float(lt), float(lg)],
                    radius=args.get("radius", 2),
                    color=colour,
                ).add_to(fmap)
        elif call["method"] == "addPolygons":
            for ring in args["data"]:
                folium.Polygon([[b, a] for a, b in ring]).add_to(fmap)
        elif call["method"] == "addPolylines":
            folium.PolyLine(
                [[float(b), float(a)] for a, b in zip(args["lng"], args["lat"], strict=True)],
                tooltip=str(args.get("label", "")),
            ).add_to(fmap)
    fmap.fit_bounds([[float(lat.min()), float(lon.min())],
                     [float(lat.max()), float(lon.max())]])
    return fmap

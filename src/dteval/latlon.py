"""Latitude/longitude helpers -- port of ``R/misc.tube.lat.lon.R``.

``tube_in_xy_polygon`` flags which tubes fall inside a polygon, typically a
Clean Air Zone boundary, so results can be split by inside/outside.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dteval.handlers import DTEvalError, check_tube_data
from dteval.rcompat.coerce import as_character_series
from dteval.rcompat.merge import r_merge
from dteval.tagging import tag_tube_required

__all__ = ["tube_in_xy_polygon"]


def tube_in_xy_polygon(
    data: pd.DataFrame,
    polygon: Any,
    output: str | list[str] = "ans",
    rename: str | list[str] = ".in_polygon",
    **kwargs,
) -> pd.DataFrame:
    """Port of ``tubeInXYPolygon``.

    Adds a logical column (``.in_polygon`` by default) saying whether each
    tube's location lies inside ``polygon``. ``output='id'`` instead adds the
    matched polygon's index as ``<rename>.id``; naming a non-geometry column of
    the polygon source copies that column across.

    Only distinct coordinate pairs are tested and the answers are merged back,
    which is what R does -- and it means the result comes back in R's merge
    order, sorted on the coordinate pair as text.
    """
    outputs = [output] if isinstance(output, str) else list(output)
    renames = [rename] if isinstance(rename, str) else list(rename)

    extra = [o for o in outputs if o not in ("ans", "id")]
    if extra and len(outputs) != len(renames):
        renames = [*renames, *extra]
    if len(outputs) != len(renames):
        raise DTEvalError("[tubeInXYPolygon] output/rename mismatch, lengths differ...")

    lat = kwargs.get("lat", ".latitude")
    lon = kwargs.get("lon", ".longitude")
    data = tag_tube_required(data, required=[lat, lon], **kwargs)

    df1 = pd.DataFrame(data).copy()
    df1 = check_tube_data(df1, [lon, lat], if_err="stop<<tubeInXYPolygon>>data lat/lon")
    df1 = df1[[lon, lat]]
    df1 = df1[~df1[lon].isna()]
    df1 = df1[~df1[lat].isna()]
    key = as_character_series(df1[lon]) + " " + as_character_series(df1[lat])
    df1 = df1[~key.duplicated()].reset_index(drop=True)

    geoms, attrs = _polygon_geometries(polygon, kwargs)
    hits = _within(df1[lon], df1[lat], geoms)

    if "ans" in outputs:
        name = renames[outputs.index("ans")]
        df1[name] = pd.Series([h is not None for h in hits], dtype="boolean")
    if "id" in outputs:
        name = renames[outputs.index("id")]
        df1[f"{name}.id"] = pd.array(
            [pd.NA if h is None else h + 1 for h in hits], dtype="Int32"
        )
    for column in attrs:
        if column in outputs:
            name = renames[outputs.index(column)]
            df1[name] = [None if h is None else attrs[column][h] for h in hits]

    if df1.shape[1] <= 2:
        import warnings

        warnings.warn(
            "[tubeInXYPolygon] nothing added/updated... maybe check inputs?", stacklevel=2
        )
        return data

    data = data[[c for c in data.columns if c not in renames]]
    return r_merge(data, df1, by=[lon, lat])


def _polygon_geometries(polygon: Any, kwargs: dict):
    """Get shapely geometries plus any non-geometry attribute columns.

    ``sf`` objects are transformed to WGS84 first; a plain frame of x/y columns
    is read with ``x``/``y``/``crs`` overrides, as R allows.
    """
    from dteval.datasets import PolygonSource

    if isinstance(polygon, PolygonSource):
        attrs = {"name": polygon.names} if any(n is not None for n in polygon.names) else {}
        return [_as_polygon(g) for g in polygon.geoms], attrs

    if isinstance(polygon, pd.DataFrame):
        from shapely.geometry import Polygon

        x = kwargs.get("x", "X")
        y = kwargs.get("y", "Y")
        if x not in polygon.columns or y not in polygon.columns:
            raise DTEvalError(
                f"[tubeInXYPolygon] Sorry, can't find/build poly x/y term '{x}'/'{y}'\n"
            )
        return [Polygon(zip(polygon[x], polygon[y], strict=True))], {}

    if hasattr(polygon, "geoms"):  # shapely multi-geometry
        return list(polygon.geoms), {}
    if hasattr(polygon, "geom_type"):  # single shapely geometry
        return [polygon], {}
    raise DTEvalError("[tubeInXYPolygon] polygon source not understood")


def _as_polygon(geom):
    """R's ``st_cast(polygon, "MULTIPOLYGON")``.

    Boundaries are often supplied as lines rather than areas -- ``caz.brd`` is
    a ``MULTILINESTRING`` -- and ``tubeInXYPolygon`` casts to ``MULTIPOLYGON``
    before testing. Without the cast a point-in-polygon test against a line is
    always false, which silently marks every tube as outside the zone.
    """
    from shapely.geometry import MultiPolygon, Polygon

    kind = geom.geom_type
    if kind in ("Polygon", "MultiPolygon"):
        return geom
    if kind == "LineString":
        return Polygon(geom.coords)
    if kind == "MultiLineString":
        return MultiPolygon([Polygon(line.coords) for line in geom.geoms])
    if kind == "LinearRing":
        return Polygon(geom)
    raise DTEvalError(
        f"[tubeInXYPolygon] cannot use a {kind} as a polygon boundary"
    )


def _within(lons, lats, geoms) -> list[int | None]:
    """``sf::st_within`` -- index of the containing polygon, or None."""
    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "tube_in_xy_polygon needs shapely. Install it with: "
            "pip install 'dteval[geo]'"
        ) from exc

    prepared = [prep(g) for g in geoms]
    out: list[int | None] = []
    for lon, lat in zip(lons, lats, strict=True):
        point = Point(float(lon), float(lat))
        match = None
        for i, geom in enumerate(prepared):
            if geom.contains(point):
                match = i
                break
        out.append(match)
    return out

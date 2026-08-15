"""Bundled DTEval datasets.

These are the upstream R datasets, converted by ``tools/export_datasets.R`` into
the lossless typed-JSON form (see :mod:`dteval.rcompat.rserial`) so that column
classes survive the trip: ``year_of_measurement`` stays an integer, the sampling
dates stay ``Date``, and nothing is inferred from text.

Each loader returns a fresh copy, so callers can tag and mutate freely without
affecting anyone else -- DTEval's functions add columns to whatever they are
given, and a shared frame would accumulate tags across calls.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from dteval.rcompat.rserial import read_rjson

__all__ = ["caz_brd", "data_path", "dt_brd", "dt_calendar"]

_DATA_DIR = Path(__file__).parent / "data"


def data_path(name: str) -> Path:
    """Absolute path to a bundled data file."""
    p = _DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"bundled dataset {name!r} is missing from {_DATA_DIR}. "
            "Regenerate with: Rscript tools/export_datasets.R"
        )
    return p


@cache
def _load_frame(name: str) -> pd.DataFrame:
    return read_rjson(data_path(f"{name}.json.gz"))


def dt_brd() -> pd.DataFrame:
    """Bradford diffusion tube data, 2020-2025 roadside-valid subset (11273x11).

    Columns follow the R dataset exactly, including the pre-tagged
    ``.start_date`` / ``.end_date`` derived from the Defra sampling calendar.
    """
    return _load_frame("dt_brd").copy()


def dt_calendar() -> pd.DataFrame:
    """The Defra/LAQM diffusion tube sampling calendar, 2015-2026 (144x5)."""
    return _load_frame("dt_calendar").copy()


@cache
def _load_caz() -> dict[str, Any]:
    with data_path("caz_brd.geojson").open() as fh:
        return json.load(fh)


def caz_brd():
    """Bradford 2022 Clean Air Zone boundary.

    Returns a ``shapely`` geometry collection wrapper with the same role the
    ``sf`` object plays in R: a polygon source for :func:`dteval.tube_in_xy_polygon`.
    Coordinates are WGS84 (EPSG:4326), matching the R object after transform.
    """
    from shapely.geometry import shape

    gj = _load_caz()
    feats = gj.get("features", [])
    geoms = [shape(f["geometry"]) for f in feats]
    names = [(f.get("properties") or {}).get("name") for f in feats]
    return PolygonSource(geoms=geoms, names=names)


class PolygonSource:
    """A minimal stand-in for an ``sf`` polygon layer.

    DTEval only ever asks a polygon source two things -- "which of these points
    are inside you?" and "what are your non-geometry columns?" -- so this carries
    just the geometries plus their attributes, and leaves CRS handling to
    :func:`dteval.tube_in_xy_polygon`.
    """

    __slots__ = ("geoms", "names", "crs")

    def __init__(self, geoms, names=None, crs: str = "EPSG:4326"):
        self.geoms = list(geoms)
        self.names = list(names) if names is not None else [None] * len(self.geoms)
        self.crs = crs

    def __len__(self) -> int:
        return len(self.geoms)

    def __repr__(self) -> str:
        return f"PolygonSource({len(self.geoms)} geometry/-ies, crs={self.crs!r})"

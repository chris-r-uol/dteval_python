"""Web Mercator projection and OpenStreetMap basemaps for tube maps.

Two separable things live here, and only one of them needs a network.

:func:`to_mercator` and :func:`from_mercator` are pure coordinate transforms on
numpy and are always available. They are not optional decoration: OSM tiles are
Web Mercator (EPSG:3857) and tube data is longitude/latitude, so drawing one on
the other without projecting misplaces the data *subtly* rather than obviously
-- the failure mode that survives review. Anything rendering a
:meth:`~dteval.plots.tube_plot.TubePlot.to_spec` payload onto tiles, in Python
or in a browser, has to do the same transform.

:func:`fetch_basemap` downloads and stitches tiles, and needs the ``basemap``
extra. ``contextily`` does this job too, but it pulls in rasterio and mercantile
for what is essentially slippy-map arithmetic, and this package has kept its
dependency list deliberately short.

OSM's tile usage policy asks for an identifying User-Agent and no bulk
downloading, so tiles are cached on disk, requests are spaced, and the zoom is
capped by tile count. Attribution is required by the Open Database Licence:
:func:`attribute` puts it on the axes and is not optional decoration either.
"""

from __future__ import annotations

import io
import math
import os
import time
from pathlib import Path

import numpy as np

__all__ = [
    "ATTRIBUTION",
    "attribute",
    "choose_zoom",
    "fetch_basemap",
    "from_mercator",
    "tile_index",
    "to_mercator",
]

#: Radius of the sphere Web Mercator is defined on (not the WGS84 ellipsoid).
R_EARTH = 6378137.0

#: Web Mercator is undefined at the poles and is conventionally cut here, at
#: the latitude that makes the projected world square.
MAX_LATITUDE = 85.05112878

TILE_PX = 256

#: Required by the Open Database Licence whenever OSM tiles or data are shown.
ATTRIBUTION = "Map data © OpenStreetMap contributors"

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = (
    "dteval/0.1 (diffusion tube figure rendering; "
    "https://github.com/chris-r-uol/dteval_python)"
)

#: Politeness cap. Bradford at zoom 13 is ~56 tiles; much beyond this is bulk
#: downloading, which OSM's tile policy asks us not to do.
MAX_TILES = 80

#: Seconds between requests that actually reach the network.
REQUEST_SPACING = 0.12


def _cache_dir() -> Path:
    override = os.environ.get("DTEVAL_TILE_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "dteval" / "tiles"


# ------------------------------------------------------------- projection ---
def to_mercator(lon, lat):
    """Project longitude/latitude in degrees to Web Mercator metres."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.clip(np.asarray(lat, dtype="float64"), -MAX_LATITUDE, MAX_LATITUDE)
    x = R_EARTH * np.deg2rad(lon)
    y = R_EARTH * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
    return x, y


def from_mercator(x, y):
    """Inverse of :func:`to_mercator`, back to degrees."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    lon = np.rad2deg(x / R_EARTH)
    lat = np.rad2deg(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)
    return lon, lat


def tile_index(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """The slippy-map tile containing a point, as ``(x, y)``."""
    n = 2**zoom
    lat = min(max(lat, -MAX_LATITUDE), MAX_LATITUDE)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` of one tile, in degrees."""
    n = 2**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def choose_zoom(bounds, max_tiles: int = MAX_TILES) -> int:
    """Highest zoom whose tile count for ``bounds`` stays within the cap.

    ``bounds`` is ``(west, south, east, north)`` in degrees.
    """
    west, south, east, north = bounds
    if max_tiles < 1:
        raise ValueError("max_tiles must be at least 1")
    for zoom in range(19, 0, -1):
        x0, y0 = tile_index(west, north, zoom)
        x1, y1 = tile_index(east, south, zoom)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            return zoom
    return 0


# ------------------------------------------------------------------ tiles ---
def _fetch_tile(x: int, y: int, zoom: int, cache_dir: Path):
    """One tile as a PIL image, from the disk cache when possible."""
    try:
        import requests
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "basemap tiles need requests and Pillow. Install them with: "
            "pip install 'dteval[basemap]'"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{zoom}_{x}_{y}.png"
    if path.exists():
        return Image.open(path).convert("RGB")

    response = requests.get(
        TILE_URL.format(z=zoom, x=x, y=y),
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    path.write_bytes(response.content)
    time.sleep(REQUEST_SPACING)
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def fetch_basemap(
    bounds,
    zoom: int | None = None,
    muted: float = 0.55,
    lighten: float = 0.28,
    max_tiles: int = MAX_TILES,
    cache_dir: str | Path | None = None,
):
    """An OSM basemap for ``bounds``, ready to draw under projected data.

    ``bounds`` is ``(west, south, east, north)`` in degrees. Returns
    ``(image, extent)`` where ``image`` is an RGB float array in [0, 1] and
    ``extent`` is ``(left, right, bottom, top)`` in **Mercator metres**, so it
    goes straight into ``ax.imshow(image, extent=extent)`` alongside data put
    through :func:`to_mercator`.

    ``muted`` desaturates and ``lighten`` washes the tiles out. Standard OSM is
    designed to be read on its own; at full strength a concentration surface
    over it fights the road colours and both lose. That is a local image
    operation, so the source and its attribution are unchanged.
    """
    west, south, east, north = bounds
    if not (west < east and south < north):
        raise ValueError("bounds must be (west, south, east, north) with west<east, south<north")

    zoom = choose_zoom(bounds, max_tiles) if zoom is None else int(zoom)
    cache = Path(cache_dir) if cache_dir is not None else _cache_dir()

    x0, y0 = tile_index(west, north, zoom)
    x1, y1 = tile_index(east, south, zoom)
    count = (x1 - x0 + 1) * (y1 - y0 + 1)
    if count > max_tiles:
        raise ValueError(
            f"zoom {zoom} needs {count} tiles for these bounds, over the {max_tiles} cap; "
            "lower the zoom or raise max_tiles deliberately"
        )

    from PIL import Image

    canvas = Image.new("RGB", ((x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            canvas.paste(_fetch_tile(x, y, zoom, cache), ((x - x0) * TILE_PX, (y - y0) * TILE_PX))

    array = np.asarray(canvas, dtype="float64") / 255.0
    grey = array.mean(axis=2, keepdims=True)
    array = array * (1 - muted) + grey * muted
    array = array * (1 - lighten) + lighten

    span = 2 * math.pi * R_EARTH / 2**zoom
    left = x0 * span - math.pi * R_EARTH
    right = (x1 + 1) * span - math.pi * R_EARTH
    top = math.pi * R_EARTH - y0 * span
    bottom = math.pi * R_EARTH - (y1 + 1) * span
    return array, (left, right, bottom, top)


def attribute(ax, size: float = 6.5, colour: str = "#3C4A44") -> None:
    """Put the ODbL-required credit on a matplotlib axes."""
    ax.text(
        0.995, 0.006, ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=size, color=colour, zorder=20,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.4},
    )

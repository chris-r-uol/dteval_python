"""Web Mercator projection and basemap tiling.

No test here touches the network. Tile fetching is monkeypatched, so the maths
that decides *where* data lands is tested independently of whether a tile
server is reachable -- which is the part that can silently be wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dteval.plots import basemap as bm


# ------------------------------------------------------------- projection ---
def test_the_origin_projects_to_the_origin():
    x, y = bm.to_mercator(0.0, 0.0)
    assert float(x) == 0.0
    assert float(y) == pytest.approx(0.0, abs=1e-9)


# Independently known EPSG:3857 coordinates. These are deliberately hard-coded
# rather than recomputed from bm.R_EARTH: an assertion derived from the module's
# own constant passes happily when that constant is wrong, and a wrong earth
# radius offsets data against the tiles by ~0.1% -- about 200 m over Bradford,
# which is exactly the kind of misregistration nobody notices.
KNOWN_MERCATOR = [
    ((180.0, 0.0), (20037508.342789244, 0.0)),
    ((0.0, 45.0), (0.0, 5621521.486192066)),
    ((-1.75, 53.8), (-194809.10888822874, 7132369.312295495)),
    ((0.0, 85.05112878), (0.0, 20037508.343038812)),
]


@pytest.mark.parametrize(("lonlat", "expected"), KNOWN_MERCATOR)
def test_projection_matches_known_epsg3857_coordinates(lonlat, expected):
    x, y = bm.to_mercator(*lonlat)
    assert float(x) == pytest.approx(expected[0], abs=1e-6)
    assert float(y) == pytest.approx(expected[1], abs=1e-6)


def test_the_earth_radius_is_the_spherical_mercator_one():
    """EPSG:3857 is defined on a sphere of exactly this radius, not on WGS84's
    ellipsoid and not on the 6371 km mean radius used elsewhere in this
    package for haversine distances."""
    assert bm.R_EARTH == 6378137.0


def test_the_projected_world_is_square():
    """The cut-off latitude is chosen to make it so; a wrong constant here
    would stretch every map vertically."""
    x, _ = bm.to_mercator(180.0, 0.0)
    _, y = bm.to_mercator(0.0, bm.MAX_LATITUDE)
    assert float(y) == pytest.approx(float(x), rel=1e-8)


@pytest.mark.parametrize("lon", [-179.9, -1.75, 0.0, 13.4, 179.9])
@pytest.mark.parametrize("lat", [-84.0, -53.8, 0.0, 53.8, 84.0])
def test_projection_round_trips(lon, lat):
    x, y = bm.to_mercator(lon, lat)
    back_lon, back_lat = bm.from_mercator(x, y)
    assert float(back_lon) == pytest.approx(lon, abs=1e-9)
    assert float(back_lat) == pytest.approx(lat, abs=1e-9)


def test_projection_is_vectorised():
    lon = np.array([-1.95, -1.70])
    lat = np.array([53.75, 53.93])
    x, y = bm.to_mercator(lon, lat)
    assert x.shape == lon.shape and y.shape == lat.shape
    assert x[0] < x[1] and y[0] < y[1]


@pytest.mark.parametrize("pole", [90.0, -90.0, 120.0, -120.0])
def test_latitude_is_clipped_at_the_mercator_cut_off(pole):
    """Mercator sends the poles to infinity.

    Asserting only isfinite() is not enough: tan(pi/2) evaluates to ~1.6e16
    rather than inf in floating point, so an unclipped implementation passes
    that check while returning a y three hundred million times too large. The
    clipped value has to equal the cut-off latitude's exactly.
    """
    _, clipped = bm.to_mercator(0.0, math.copysign(bm.MAX_LATITUDE, pole))
    _, got = bm.to_mercator(0.0, pole)
    assert float(got) == pytest.approx(float(clipped), rel=1e-12)
    assert abs(float(got)) < 2.1e7


# ------------------------------------------------------------------ tiles ---
def test_zoom_zero_is_a_single_tile():
    assert bm.tile_index(-1.75, 53.8, 0) == (0, 0)


def test_tile_indices_split_at_the_origin():
    assert bm.tile_index(0.1, -0.1, 1) == (1, 1)
    assert bm.tile_index(-0.1, 0.1, 1) == (0, 0)


@pytest.mark.parametrize("zoom", [2, 8, 13, 16])
@pytest.mark.parametrize(("lon", "lat"), [(-1.75, 53.8), (0.0, 0.0), (13.377, 52.518), (151.2, -33.9)])
def test_the_returned_tile_actually_contains_the_point(lon, lat, zoom):
    """Stronger than pinning magic tile numbers: whatever tile comes back, the
    point has to lie inside its own bounds."""
    x, y = bm.tile_index(lon, lat, zoom)
    west, south, east, north = bm.tile_bounds(x, y, zoom)
    assert west <= lon <= east
    assert south <= lat <= north


def test_choose_zoom_respects_the_tile_cap():
    bounds = (-1.9506, 53.7528, -1.6952, 53.9338)  # the dt.brd extent
    for cap in (4, 16, 80, 400):
        zoom = bm.choose_zoom(bounds, max_tiles=cap)
        x0, y0 = bm.tile_index(bounds[0], bounds[3], zoom)
        x1, y1 = bm.tile_index(bounds[2], bounds[1], zoom)
        assert (x1 - x0 + 1) * (y1 - y0 + 1) <= cap


def test_choose_zoom_picks_the_highest_zoom_that_fits():
    bounds = (-1.9506, 53.7528, -1.6952, 53.9338)
    zoom = bm.choose_zoom(bounds, max_tiles=80)
    x0, y0 = bm.tile_index(bounds[0], bounds[3], zoom + 1)
    x1, y1 = bm.tile_index(bounds[2], bounds[1], zoom + 1)
    assert (x1 - x0 + 1) * (y1 - y0 + 1) > 80, "one zoom further should not fit"


def test_a_bad_tile_cap_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        bm.choose_zoom((-1.0, 53.0, -0.9, 53.1), max_tiles=0)


# --------------------------------------------------------------- stitching ---
@pytest.fixture
def fake_tiles(monkeypatch):
    """Serve solid-colour tiles so stitching can be checked without a network."""
    pytest.importorskip("PIL")
    from PIL import Image

    served = []

    def _fake(x, y, zoom, cache_dir):
        served.append((x, y, zoom))
        # a per-tile shade, so a mis-ordered paste is visible in the output
        return Image.new("RGB", (bm.TILE_PX, bm.TILE_PX), (x % 256, y % 256, zoom))

    monkeypatch.setattr(bm, "_fetch_tile", _fake)
    return served


def test_the_stitched_extent_contains_the_requested_bounds(fake_tiles):
    """The mosaic is whole tiles, so it must cover the request and may overhang
    -- but it must never fall short, or data would be drawn off the map."""
    bounds = (-1.9506, 53.7528, -1.6952, 53.9338)
    image, (left, right, bottom, top) = bm.fetch_basemap(bounds, zoom=12)

    (wx, ex), (sy, ny) = bm.to_mercator(
        [bounds[0], bounds[2]], [bounds[1], bounds[3]]
    )
    assert left <= wx and right >= ex
    assert bottom <= sy and top >= ny
    assert image.ndim == 3 and image.shape[2] == 3


def test_the_stitched_image_matches_the_tile_grid(fake_tiles):
    bounds = (-1.9506, 53.7528, -1.6952, 53.9338)
    zoom = 12
    image, _ = bm.fetch_basemap(bounds, zoom=zoom)
    x0, y0 = bm.tile_index(bounds[0], bounds[3], zoom)
    x1, y1 = bm.tile_index(bounds[2], bounds[1], zoom)
    assert image.shape[0] == (y1 - y0 + 1) * bm.TILE_PX
    assert image.shape[1] == (x1 - x0 + 1) * bm.TILE_PX
    assert len(fake_tiles) == (x1 - x0 + 1) * (y1 - y0 + 1)


def test_the_extent_is_square_per_tile(fake_tiles):
    """A tile is square in Mercator metres. If it is not here, the aspect
    ratio of every map built on it is wrong."""
    image, (left, right, bottom, top) = bm.fetch_basemap(
        (-1.9506, 53.7528, -1.6952, 53.9338), zoom=12
    )
    per_px_x = (right - left) / image.shape[1]
    per_px_y = (top - bottom) / image.shape[0]
    assert per_px_x == pytest.approx(per_px_y, rel=1e-12)


def test_tiles_are_muted_towards_grey_and_light(fake_tiles):
    vivid, _ = bm.fetch_basemap((-1.95, 53.75, -1.70, 53.93), zoom=12,
                                muted=0.0, lighten=0.0)
    washed, _ = bm.fetch_basemap((-1.95, 53.75, -1.70, 53.93), zoom=12,
                                 muted=1.0, lighten=0.5)
    assert washed.mean() > vivid.mean(), "lighten should raise overall brightness"
    channel_spread = washed.max(axis=2) - washed.min(axis=2)
    assert channel_spread.max() == pytest.approx(0.0, abs=1e-12), "muted=1 is greyscale"
    assert (washed >= 0).all() and (washed <= 1).all()


def test_going_over_the_tile_cap_is_refused_not_silently_obeyed(fake_tiles):
    with pytest.raises(ValueError, match="over the .* cap"):
        bm.fetch_basemap((-2.5, 53.0, -1.0, 54.5), zoom=14, max_tiles=20)


def test_nonsense_bounds_are_rejected(fake_tiles):
    with pytest.raises(ValueError, match="west<east"):
        bm.fetch_basemap((-1.0, 53.9, -1.5, 53.7), zoom=10)


# ---------------------------------------------------------------- licence ---
def test_the_attribution_names_openstreetmap():
    """ODbL requires it, so it is not a stylistic choice."""
    assert "OpenStreetMap" in bm.ATTRIBUTION


def test_attribute_draws_the_credit_on_the_axes():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    try:
        bm.attribute(ax)
        assert any(bm.ATTRIBUTION in t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)


def test_requests_identify_themselves():
    """OSM's tile policy asks for a User-Agent that says who is calling."""
    assert "dteval" in bm.USER_AGENT
    assert "http" in bm.USER_AGENT

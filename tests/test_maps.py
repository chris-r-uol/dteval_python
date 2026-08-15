"""Map behaviour that the parity fixtures do not reach.

The fixtures pin ``tube_map`` and ``leaflet_tube_map`` against R for the cases
the R package can actually run here. What they cannot cover is the basemap
itself -- R fetches it through a Java tile stack that is not installed -- and
the JSON a web frontend consumes. Both are checked directly instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import dteval as dte
from dteval.plots.maps import _expand, _pretty, colour_numeric


@pytest.fixture(scope="module")
def tubes():
    return dte.datasets.dt_brd()


def test_expand_grows_a_range_symmetrically():
    assert _expand(0.0, 10.0, 0.05) == (-0.5, 10.5)
    assert _expand(-2.0, -1.0, 0.1) == pytest.approx((-2.1, -0.9))


def test_map_extent_covers_every_tube(tubes):
    """The 5% border must leave every sampling point inside the panel."""
    plot = dte.tube_map(tubes)
    tagged = dte.tag_tube(tubes)
    for column in (".longitude", ".latitude"):
        low, high = plot.limits[column]
        assert low < np.nanmin(tagged[column])
        assert high > np.nanmax(tagged[column])


def test_map_requests_a_basemap_it_does_not_fetch(tubes):
    """tube_map hands the client a tile request rather than pixels."""
    spec = dte.tube_map(tubes).to_spec()
    assert spec["basemap"]["provider"] == "esri"
    assert spec["basemap"]["bbox"] == spec["limits"]
    assert spec["coord"] == "coord_quickmap"
    assert spec["blank_axes"] is True
    assert not any(layer["geom"] == "GeomRaster" for layer in spec["layers"])


def test_map_spec_is_json_serialisable(tubes):
    json.loads(dte.tube_map(tubes, facet=".year").to_json())


def test_leaflet_colours_interpolate_in_lab_space():
    """leaflet's colorNumeric goes through scales::colour_ramp, which
    interpolates in CIE Lab. sRGB interpolation is off by a level or two."""
    # The three values R produces for the first dt.brd tubes; see the
    # map/leafletTubeMap.coloured fixture.
    colours = colour_numeric([0.0, 0.5, 1.0], "Spectral")
    assert colours[0] == "#9E0142"
    assert colours[-1] == "#5E4FA2"
    assert colours[1] == "#FFFFBF"


def test_leaflet_colours_handle_a_constant_column():
    assert colour_numeric([7.0, 7.0, 7.0]) == ["#9E0142"] * 3


def test_leaflet_marker_count_matches_the_data(tubes):
    spec = dte.leaflet_tube_map(tubes).to_spec()
    tiles, markers = spec["calls"]
    assert tiles["method"] == "addProviderTiles"
    assert markers["method"] == "addCircleMarkers"
    assert len(markers["args"]["lng"]) == len(tubes)
    assert markers["args"]["color"] == "#03F"


def test_leaflet_prefixed_argument_implies_its_layer(tubes):
    """`point.color=` alone is enough to ask for the point layer, as in R."""
    spec = dte.leaflet_tube_map(tubes, **{"point.color": ".value"})
    methods = [call["method"] for call in spec.calls]
    assert methods == ["addProviderTiles", "addCircleMarkers"]
    assert len(spec.calls[1]["args"]["color"]) == len(tubes)


def test_leaflet_none_draws_no_layers(tubes):
    spec = dte.leaflet_tube_map(tubes, plot_type="none")
    assert [call["method"] for call in spec.calls] == ["addProviderTiles"]


def test_leaflet_map_is_extensible(tubes):
    """Passing a map back in adds to it rather than starting over, which is how
    R's `leafletTubeMap(previous, ...)` behaves."""
    first = dte.leaflet_tube_map(tubes.head(50))
    second = dte.leaflet_tube_map(first, **{"point.radius": 5})
    assert [call["method"] for call in second.calls] == [
        "addProviderTiles",
        "addCircleMarkers",
        "addCircleMarkers",
    ]


def test_leaflet_spec_is_json_serialisable(tubes):
    json.loads(dte.leaflet_tube_map(tubes, **{"point.color": ".value"}).to_json())


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 100], [0, 20, 40, 60, 80, 100]),
        ([2.3, 17.8], [0, 5, 10, 15, 20]),
        ([1.2, 3.7], [1, 1.5, 2, 2.5, 3, 3.5, 4]),
        ([0.001, 0.009], [0, 0.002, 0.004, 0.006, 0.008, 0.01]),
    ],
)
def test_pretty_matches_r(values, expected):
    """R's pretty() picks a 1/2/5 x 10^k step covering the data. Expectations
    are R's own output for these inputs."""
    assert _pretty(values) == pytest.approx(expected)


def test_pretty_on_a_flat_range_is_a_noted_difference():
    """R invents a span (pretty(c(5, 5)) is 0 5); we keep the one break."""
    assert _pretty([5, 5]) == [5.0]

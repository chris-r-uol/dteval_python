"""dteval -- Python port of the DTEval R package.

Diffusion tube (DT) air quality data pre-processing, analysis and evaluation.

The R package at https://github.com/karlropkins/DTEval defines correct
behaviour; this port reproduces its values, and the parity suite enforces that.

R's camelCase names become snake_case here (``tagTubeStartEnd`` ->
``tag_tube_start_end``). Tagged column names such as ``.value``,
``.start_date`` and ``.sample_id`` are unchanged, because they are part of the
data contract rather than the API surface.
"""

from __future__ import annotations

__version__ = "0.1.1.3"

from dteval import datasets, plots, rcompat
from dteval.accuracy import test_tube_accuracy
from dteval.annual import tube_annual_cover, tube_annual_test
from dteval.calc import calc_tube_stat
from dteval.cluster import cluster_tube_data
from dteval.deseason import deseason_tube_data
from dteval.fit import fit_tube_model, fit_tube_model_gam, fit_tube_model_loess
from dteval.handlers import DTEvalError, check_tube_data, get_tube_x
from dteval.latlon import tube_in_xy_polygon
from dteval.loess import r_loess
from dteval.meta import (
    add_tube_meta,
    check_tube_meta,
    extract_and_add_tube_meta,
    extract_tube_meta,
    pad_tube_meta,
    repair_tube_meta,
)
from dteval.plots import (
    LeafletMap,
    TubePlot,
    leaflet_tube_map,
    test_tube_meta,
    tube_map,
    tube_plot,
    tube_time_plot,
)
from dteval.precision import test_tube_precision
from dteval.summaries import tube_summary, tube_summary_lat_lon, tube_summary_sample
from dteval.tagging import (
    tag_tube,
    tag_tube_date,
    tag_tube_lat_lon,
    tag_tube_location,
    tag_tube_month,
    tag_tube_required,
    tag_tube_sample_id,
    tag_tube_start_end,
    tag_tube_value,
    tag_tube_year,
)

__all__ = [
    "DTEvalError",
    "__version__",
    "add_tube_meta",
    "calc_tube_stat",
    "check_tube_meta",
    "check_tube_data",
    "cluster_tube_data",
    "datasets",
    "deseason_tube_data",
    "TubePlot",
    "extract_and_add_tube_meta",
    "extract_tube_meta",
    "fit_tube_model",
    "fit_tube_model_gam",
    "fit_tube_model_loess",
    "get_tube_x",
    "plots",
    "r_loess",
    "pad_tube_meta",
    "rcompat",
    "repair_tube_meta",
    "tag_tube",
    "tag_tube_date",
    "tag_tube_lat_lon",
    "tag_tube_location",
    "tag_tube_month",
    "tag_tube_required",
    "tag_tube_sample_id",
    "tag_tube_start_end",
    "tag_tube_value",
    "tag_tube_year",
    "LeafletMap",
    "leaflet_tube_map",
    "test_tube_accuracy",
    "test_tube_meta",
    "test_tube_precision",
    "tube_annual_cover",
    "tube_annual_test",
    "tube_in_xy_polygon",
    "tube_map",
    "tube_plot",
    "tube_summary",
    "tube_summary_lat_lon",
    "tube_summary_sample",
    "tube_time_plot",
]

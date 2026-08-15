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
from dteval.annual import tube_annual_cover
from dteval.calc import calc_tube_stat
from dteval.handlers import DTEvalError, check_tube_data, get_tube_x
from dteval.latlon import tube_in_xy_polygon
from dteval.loess import r_loess
from dteval.plots import TubePlot, tube_plot, tube_time_plot
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
    "calc_tube_stat",
    "check_tube_data",
    "datasets",
    "TubePlot",
    "get_tube_x",
    "plots",
    "r_loess",
    "rcompat",
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
    "test_tube_precision",
    "tube_annual_cover",
    "tube_in_xy_polygon",
    "tube_plot",
    "tube_summary",
    "tube_summary_lat_lon",
    "tube_summary_sample",
    "tube_time_plot",
]

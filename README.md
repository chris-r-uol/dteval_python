# dteval — Python port of the DTEval R package

[![CI](https://github.com/chris-r-uol/dteval_python/actions/workflows/ci.yml/badge.svg)](https://github.com/chris-r-uol/dteval_python/actions/workflows/ci.yml)

A Python port of [DTEval](https://github.com/karlropkins/DTEval), Karl Ropkins'
R package for pre-processing, analysis and evaluation of diffusion tube (DT)
data collected in air quality assessment exercises.

## The R package is the gold standard

This port exists to produce **the same numbers as the R package**. Any
divergence is a bug here, not a difference of opinion. That is enforced, not
merely intended:

- `parity/manifest.yaml` declares each comparison case once, giving the R
  expression and the Python expression side by side.
- `parity/generate.R` runs the R side against the pinned upstream tree and
  serialises the results losslessly (17-significant-digit doubles, explicit
  column classes, factor levels, NA-vs-NaN).
- `pytest -m parity` compares the Python results against those fixtures.
  Numbers are held to a relative 1e-6; **structure is exact** — column names
  and order, row order, dtypes, factor levels, NA-vs-NaN, and every integer,
  string, date and boolean. Any looser bound needs a documented reason in the
  manifest.
- CI additionally regenerates the fixtures from live R on every PR and fails if
  they drift, so the reference values cannot silently go stale.

Output conventions follow the R package: tagged column names (`.value`,
`.start_date`, `.sample_id`, `..pred`), the `{"data":…, "plot":…, "lookup":…,
"report":…}` return shape, the report strings, and the plotting argument system.

## Install

```bash
pip install dteval            # core: numpy, pandas, scipy
pip install "dteval[all]"     # + shapely, aqeval, plotnine
```

The core install has no compiled dependency beyond numpy/pandas/scipy and can
run every analysis in the package. Optional extras cover point-in-polygon
(`geo`), nearest-site distances (`aqeval`) and server-side figure rendering
(`plots`); calling something without its extra tells you which to install. See
[`docs/backend.md`](docs/backend.md) if you are deploying this behind an API.

## Quick start

```python
import dteval as dte

dt = dte.datasets.dt_brd()

dte.calc_tube_stat(dt)
dte.tube_summary(dt)

res = dte.test_tube_precision(dt)
print(res["report"])
#> '.value' (rep = 3 subset):
#>   |all| mean: 25.69 (6.522 to 56.88) precision: -14.01[%] to 14.01[%]

res["plot"].to_spec()      # JSON-serialisable figure, no plotting stack needed
```

## Naming

R's camelCase becomes snake_case: `tagTubeStartEnd` → `tag_tube_start_end`,
`calcTubeStat` → `calc_tube_stat`. Tagged *column* names are unchanged, because
they are part of the data contract.

Note that `test_tube_precision`, `test_tube_accuracy` and `test_tube_meta` are
library functions, not tests. Reach them through the module
(`dteval.test_tube_precision(...)`) rather than importing the name directly, or
pytest will try to collect them.

## Running the parity suite

```bash
pytest -m parity
```

To regenerate the fixtures from R (requires R; `make fixtures` clones the
pinned upstream SHA and builds the dependency shims first):

```bash
make fixtures
```

To check regenerated fixtures against the committed ones, use the parity
contract rather than a byte diff:

```bash
python parity/compare_fixtures.py <committed-copy> parity/fixtures
```

`git diff parity/fixtures/` is worth a look but is not the test. Byte identity
of 17-significant-digit doubles also requires two machines to agree on
floating-point accumulation, and R sums in `LDOUBLE` — so the same R source on
the same data differs in the last ulp between a build with
`capabilities("long.double")` and one without. `compare_fixtures.py` applies
what the project actually claims: structure exactly, numbers to 1e-6.

## Known parity gaps

All 37 exported functions are ported. Three diverge from R numerically, each
documented in [`docs/parity.md`](docs/parity.md), marked non-gating in the
manifest, and with its measured deviation pinned by tests so it cannot drift
unnoticed:

| Area | Reason |
|---|---|
| `deseason_tube_data` | R fits LOESS with its default kd-tree *approximation*; we compute the exact local regression. Our fit matches R's own `surface="direct"` to 2e-13 |
| `cluster_tube_data` | R uses `clara` (PAM on random subsamples, RNG-dependent); we run PAM on the full data. Ours reproduces R's own `pam()` exactly |
| `fit_tube_model_gam` | mgcv's construction is reproduced; its smoothing-parameter optimiser is not. The GCV objective is flat near the minimum — median 0.033, max 1.32 µg/m³ |

The first two are cases where the port computes the exact answer R's shortcut
is approximating, with the measurements to show it.

The maps are gated on everything except the basemap: `tube_map` hands the
client a tile *request* rather than fetching rasters through R's Java tile
stack, and `leaflet_tube_map` is compared on layer content — 11,273 marker
positions and colours, exactly — rather than on HTML. Everything else is gated
at 1e-6 with structure exact.

## Licence

GPL-3.0-or-later, matching the upstream R package.

Upstream: Karl Ropkins, University of Leeds — <k.ropkins@its.leeds.ac.uk>

# dteval — Python port of the DTEval R package

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

To regenerate the fixtures from R (requires R plus the upstream dependencies):

```bash
make r-reference        # clone the pinned upstream SHA
Rscript parity/generate.R
git diff --exit-code parity/fixtures/
```

## Known parity gaps

Three areas cannot be reproduced bit-exactly in pure Python; they are documented
in [`docs/parity.md`](docs/parity.md), marked non-gating in the manifest, and
their measured deviation is tracked so it cannot drift unnoticed:

| Area | Reason |
|---|---|
| `deseason_tube_data` | R fits LOESS with its default kd-tree *approximation*; we compute the exact local regression. Our fit matches R's own `surface="direct"` to 2e-13 |
| `cluster_tube_data` | R uses `clara` (PAM on random subsamples, RNG-dependent); we run PAM on the full data. Ours reproduces R's own `pam()` exactly |
| `tube_map` / `leaflet_tube_map` | not yet ported |

Both gaps are cases where the port computes the exact answer R's shortcut is
approximating — with the measurements to show it. Everything else is gated.

## Licence

GPL-3.0-or-later, matching the upstream R package.

Upstream: Karl Ropkins, University of Leeds — <k.ropkins@its.leeds.ac.uk>

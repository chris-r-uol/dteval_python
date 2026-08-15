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
  Doubles must match **bit for bit** unless the manifest records a tolerance
  *and* a reason for it.
- CI additionally regenerates the fixtures from live R on every PR and fails if
  they drift, so the reference values cannot silently go stale.

Output conventions follow the R package: tagged column names (`.value`,
`.start_date`, `.sample_id`, `..pred`), the `{"data":…, "plot":…, "lookup":…,
"report":…}` return shape, the report strings, and the plotting argument system.

## Install

```bash
pip install -e ".[all]"
```

## Quick start

```python
import dteval as dte

dt = dte.datasets.dt_brd()

dte.calc_tube_stat(dt)
dte.tube_summary(dt)

res = dte.test_tube_precision(dt)
print(res["report"])
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
| `fit_tube_model_gam` | `mgcv::gam` tensor smooths with GCV/REML selection have no faithful Python equivalent |
| `tube_map` | basemap rasters come from a different tile stack; only the ggplot layers and bbox are compared |
| `leaflet_tube_map` | folium and leaflet emit structurally different HTML; compared on layer contents |

Everything else is expected to be bit-exact, and CI enforces that.

## Licence

GPL-3.0-or-later, matching the upstream R package.

Upstream: Karl Ropkins, University of Leeds — <k.ropkins@its.leeds.ac.uk>

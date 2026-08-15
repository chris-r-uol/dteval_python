# Parity with the R package

The R package is the definition of correct. This document records how that is
enforced, and — honestly — the places where bit-exactness is not achievable,
with the measured size of the gap.

## How parity is enforced

| Layer | Mechanism |
|---|---|
| Case definitions | `parity/manifest.yaml` — R expression and Python expression side by side, one file |
| R side | `parity/generate.R` sources the pinned upstream tree and serialises results |
| Serialisation | `parity/serialize.R` — `%.17g` doubles (lossless), explicit R classes, factor levels, NA-vs-NaN |
| Python side | `tests/test_parity.py` evaluates the Python expression and compares |
| Comparison | `parity/compare.py` — exact by default; a tolerance requires a documented `reason` |
| Staleness | `parity/fixtures/_lock.json` records the upstream SHA and R-source hash; CI regenerates and fails on drift |

Doubles must match **bit for bit**. `%.17g` round-trips every IEEE double, so
"the same number" means the same 64 bits, not "close enough".

## Pinned environment

The gold standard is a *specific* R build, not "R in general". Several results
below depend on how R's C sources were compiled.

- Upstream DTEval: `08a6cbda6d5703ef564f076f5ae5aafc1d7ad6d6` (2026-06-30)
- R 4.6.1, `aarch64-apple-darwin20`, conda-forge build
- `capabilities("long.double")` is `FALSE` — `LDOUBLE` is 64-bit, so R's
  accumulations are plain double arithmetic
- Locale pinned to `LC_COLLATE=C`, `LC_TIME=C`, `LC_NUMERIC=C`

### R-side dependency shims

DTEval reaches `AQEval` for exactly three calls — `findNearLatLon` (in
`testTubeAccuracy`, `deseasonTubeData(method = 2)` and `tubeSummaryLatLon`) and
`calcDateRangeStat` (in `testTubeAccuracy`). Installing the real AQEval pulls in
`loa` → `OpenStreetMap` → `rJava` → a JDK, none of which those two functions
use, and none of which is available on every machine.

`tools/build_r_shims.R` therefore builds minimal `loa` and `AQEval` packages
containing only those functions. The bodies are **not** hand-transcribed: they
are parsed out of the genuine upstream sources at a pinned commit and re-emitted
verbatim with `deparse()`, so the shim cannot drift from the real package by a
typo, and the build fails loudly if a function is renamed upstream. Provenance
is recorded in `parity/fixtures/_shims.json` and folded into `_lock.json`.

The shims install into the project-local `.Rlib` only. Delete `.Rlib` and the
real packages take over if they are ever installed.

`OpenStreetMap` is a separate matter: it is only used for `tubeMap` basemap
tiles, which are not parity-tested, so when it is unavailable it is relaxed out
of DTEval's `Imports` at install time. That is safe because the package reaches
it only through `::`, never through its `NAMESPACE`. The relaxation is recorded
in `_lock.json`.

### AQEval on the Python side

The Python port depends on the user's own
[aqeval_python](https://github.com/chris-r-uol/aqeval_python) rather than
reimplementing `find_near_lat_lon` / `calc_date_range_stat`.

Its Haversine originally agreed with R to ~1e-10 m but not bit-for-bit: over
1,884 real site pairs from `dt.brd`, 1868 distances differed. Two
floating-point details, not a difference in formula — R converts the coordinate
*difference* from degrees, and associates the second term as
`sin(dLon/2) * sin(dLon/2) * cos(lat0) * cos(lat1)`. Corrected upstream (branch
`fix/haversine-r-parity`), all 1,884 now match exactly. `tubeSummaryLatLon`
returns `distance.m` as an output value, so this had to be exact rather than
merely close.

## Findings that shaped the port

These are not incidental; each one silently changes output values if ignored.

### `as.character(double)` uses 15 significant digits

Not shortest-round-trip like Python's `repr`. This is load-bearing rather than
cosmetic: `tagTubeSampleID` builds a grouping key by pasting latitude,
longitude and dates together and takes `as.numeric(factor(...))`, so a
one-digit difference in the string changes the grouping and shifts every
`.sample_id`. `tagTubeLocation` builds `.location` the same way.

Ported in `rcompat/numfmt.py` from R's `scientific()` / `formatReal()` /
`EncodeRealDrop0()`. Verified exact on 67,454 values including every value in
`dt.brd`.

### Collation order is a value, not a presentation detail

`factor()` levels are `sort(unique(x))` under the current collation, and
`.sample_id` is the 1-based level index. Row order of `testTubePrecision` and
`testTubeAccuracy` output likewise comes from `sort(unique(data$.cut))`.

The port reproduces **R under `LC_COLLATE=C`**. A user running R under a
UK/US locale, where collation largely ignores punctuation at the primary
strength, can legitimately get different `.sample_id` numbering than this port
produces. Supporting full ICU collation is possible but would add a heavy
dependency; raise an issue if you need it.

### R `Date` is a double, and `.date` rounds half-to-even

`tagTubeDate`'s default method is `.start_date + (.end_date - .start_date)/2`.
The right-hand side is a `difftime`, and `+.Date` passes it through
`coerceTimeUnit`, which applies `round()` — half to **even**. So the result is
always a whole day, but not the one naive arithmetic gives: a 35-day period
adds 18 (17.5 rounds up to even), a 33-day period adds 16 (16.5 rounds *down*
to even).

R's `Date` is nonetheless a double, and `format()` on a fractional one
**truncates** rather than rounds (19000.5 and 19000.0 both render as the same
day; −0.5 renders as 1969-12-31). Dates are held here as `datetime64[ns]`,
which represents a fractional day exactly, and both behaviours are implemented
on top.

The parity serialiser writes a Date's underlying numeric rather than
`%Y-%m-%d`, so that any fractional component would show up as a difference
instead of being flattened away by formatting.

### R's `mean` is a two-pass algorithm

`mean(x)` is not `sum(x)/n`. R computes that, then corrects it:
`s = sum(x)/n; if finite: s += sum(x - s)/n`. Skipping the correction changes
the last bit of nearly every group mean. numpy's `sum` also uses pairwise
summation where R uses a sequential loop, so `cumsum` is used to force
sequential accumulation.

### R's compiler contracts multiply-adds into FMAs

This one is only discoverable by measurement. R's variance loop is
`sum += (x[k] - xm) * (x[k] - xm)`, and the C compiler fuses it into a single
multiply-add under the default `-ffp-contract=on`. An FMA rounds once where a
separate multiply and add round twice.

Measured over the 2,875 co-located replicate groups in `dt.brd` — the domain
`testTubePrecision` actually operates on:

| accumulation | matches R |
|---|---|
| FMA (what we do) | **2875 / 2875** |
| plain sequential | 2262 / 2875 |

The same contraction appears in `qnorm`'s Horner evaluation and in
`r = .180625 - q*q`; reproducing it there took qnorm from 5 mismatches to 0.

## Known gaps

Every gap below is marked non-gating in `parity/manifest.yaml` and has a test
that pins its measured size, so it cannot quietly widen.

### `var` / `sd` outside DTEval's domain — up to 1 ulp

On an adversarial synthetic set (n up to 60, values spanning 13 orders of
magnitude), FMA accumulation matches 312/400 and plain sequential 367/400, with
6 samples matching neither — so no single accumulation model reproduces this R
build there. At that size the compiled loop evidently takes a different shape.

**Impact:** none on DTEval's own use, which is replicate groups of n = 2/3/5
with values of similar magnitude, where the port is exact.

### `qt` general branch — up to 5 ulps

R refines Hill's expansion with a Newton step driven by `pt`/`dt`. Porting
those would mean porting R's incomplete beta as well; scipy supplies them, and
its ~1e-16 relative difference lands in the last bits.

**Exact** for `df ≈ 1` (Cauchy), `df ≈ 2`, and `df > 1e20` — all closed forms.
DTEval calls `qt(0.025, df = n - 1, lower.tail = FALSE)` where `n` is the
replicate count, so the default `n = 3` gives `df = 2` and is exact. Worst
measured deviation elsewhere is 5 ulps at `df = 1.5`.

### `tube_in_xy_polygon` — row order within a coordinate group

Values, columns and column order are exact. Only the relative order of rows
sharing an *identical* coordinate pair is not reproduced, so the comparison
canonicalises row order first (`row_order_artifact` in the manifest).

R's `merge()` is stable for small inputs but not at this scale: one 11,273-row
coordinate group comes back as rows 9550, 9549, 3392, 195, … because
`merge.data.frame`'s within-tie order comes from the C `do_merge` index rather
than from a stable sort. That is an internal artifact, not a documented output,
and since every tube shares its location with its own replicates nothing
downstream depends on it — callers group or sort anyway. Reproducing it would
mean porting R's C merge.

Worth noting the canonicalisation *found* a real bug rather than hiding one:
with row-order noise removed, `in.caz` disagreed on 7,675 rows. The cause was
that `caz.brd` is a `MULTILINESTRING` and `tubeInXYPolygon` casts it with
`st_cast(..., "MULTIPOLYGON")` before testing — without that cast a
point-in-polygon test against a *line* is always false, silently marking every
tube as outside the zone.

### `tube_summary_lat_lon` — temporary tolerance on `distance.m`

`distance.m` comes from `aqeval.find_near_lat_lon`. The currently installed
build groups the Haversine terms differently from R AQEval, giving a worst-case
disagreement of 6.5e-10 m over 1,884 real site pairs. The fix
(`aqeval_python` branch `fix/haversine-r-parity`) makes all 1,884 exact.

**This tolerance should be deleted once that fix is installed**, at which point
the case is bit-exact. Row order and every other column already are.

### `fit_tube_model_gam` — not bit-exact

`mgcv::gam` with `te()` tensor smooths and GCV/REML smoothing-parameter
selection has no faithful Python equivalent, and the native-only constraint
rules out calling R. Implemented natively; structurally correct and numerically
close, but not bit-exact. Parity is recorded and reported, not enforced.

### `tube_map` — basemap not compared

`OpenStreetMap::openmap` fetches raster tiles from a different provider stack
than `contextily`. Only the non-raster plot layers, the bounding box and the
projection are parity-tested.

### `leaflet_tube_map` — compared structurally

folium and leaflet emit different HTML. Compared on layer contents — marker
coordinates, palettes, popups, bounds — rather than output markup.

## Figures

Plots are compared on **the decisions DTEval makes**, not on pixels and not on
ggplot2's internals. For each layer the fixture records:

- the geom (`GeomPoint`, `GeomRibbon`, …);
- the **input data frame** handed to that layer — compared bit-exactly;
- which columns are mapped to which aesthetics;
- constant aesthetics (`fill = "grey"`, `colour = "red"`);

plus the plot-level labels, facet specification and palette.

This is deliberately *not* `ggplot_build()` output. Comparing that would pit
ggplot2's internal columns (`PANEL`, `group`, `flipped_aes`, computed stat
columns) against plotnine's — two libraries' implementation details rather than
DTEval's behaviour. The layer *input* data is the better contract precisely
because everything DTEval computes for itself already lives there: the band
quantiles from `calcTubeStat`, the fitted values from `fitTubeModel`, the LOESS
bounds in `testTubePrecision`. A light `built_summary` (layer row counts, panel
count) is also recorded so a structural change still shows up.

### Rendering

`TubePlot.draw()` renders via [plotnine](https://plotnine.org), which implements
the same grammar, so the conventions carry across: `theme_bw`, transparent facet
strips, the `grey` boxplot fill, and ggplot2's evenly spaced HCL hue palette
(reproduced from `grDevices::hcl` and verified exact — see
`rcompat/palette.py`).

Two rendering details are approximations of ggplot2's *presentation* rather than
of any DTEval value, and are not part of the parity contract:

- **Date axis labels.** ggplot2 picks a format from the break spacing, so a
  multi-year axis reads `2022 2023 2024`. plotnine defaults to full dates, so
  the format is chosen from the data range instead.
- **Markdown labels.** `dte_quickText` emits HTML for
  `ggtext::element_markdown` (`NO<sub>2</sub>`, `&mu;g.m<sup>-3</sup>`). The
  *label strings* are compared exactly; plotnine has no markdown renderer, so at
  draw time they are unwrapped to their nearest plain-text form (`NO2`,
  `μg.m^-3`) rather than shown as literal tags.

Side-by-side PNGs from both languages live in `docs/figures/` (Python) and
`docs/figures/R/` (R) for human review; they are not gated.

## Reproducing

```bash
make r-reference            # clone the pinned upstream SHA
Rscript parity/generate_rcompat.R
Rscript parity/generate.R
git diff --exit-code parity/fixtures/
pytest -m parity
```

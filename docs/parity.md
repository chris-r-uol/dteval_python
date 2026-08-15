# Parity with the R package

The R package is the definition of correct. This document records how that is
enforced, where the port deliberately diverges, and the measured size of every
gap.

## How parity is enforced

| Layer | Mechanism |
|---|---|
| Case definitions | `parity/manifest.yaml` — R expression and Python expression side by side, one file |
| R side | `parity/generate.R` sources the pinned upstream tree and serialises results |
| Serialisation | `parity/serialize.R` — `%.17g` doubles (lossless), explicit R classes, factor levels, NA-vs-NaN |
| Python side | `tests/test_parity.py` evaluates the Python expression and compares |
| Comparison | `parity/compare.py` — floats to 1e-6, structure exact; any override requires a documented `reason` |
| Staleness | `parity/fixtures/_lock.json` records the upstream SHA and R-source hash; `tests/test_lock.py` gates both, and CI regenerates from live R on every PR and re-checks with `parity/compare_fixtures.py` |

**Numbers are compared to a relative tolerance of 1e-6; structure is compared
exactly.**

That split is deliberate. Diffusion tube NO₂ measurements carry roughly a **10%
error bar**, so agreeing with R to the last bit is about 14 orders of margin on
something the instrument cannot resolve — and chasing it costs real complexity:
transliterated Fortran, Python-loop accumulation, and floating-point behaviour
coupled to the platform. 1e-6 sits ~4 orders inside the measurement error and
~3 orders outside any realistic bug (a wrong formula, subset or grouping shows
up at 1e-3 or larger).

What is **not** relaxed is anything that changes the answer rather than its
last digits:

- column names and order, row count and row order, dtypes
- factor levels *and their order*
- the NA-vs-NaN distinction
- every integer, string, date and boolean — including the report strings

About **56% of compared cells are non-float**, so no tolerance touches them at
all.

An earlier revision did enforce bit-exactness. That is what surfaced R's
two-pass `mean`, the FMA contraction in its `var` loop, and the need for
hand-ported `qnorm`/`qt` — roughly 520 lines of transliteration, since removed.
The structural findings it produced (below) were kept; they matter at any
tolerance.

### Why the CI gate is not `git diff`

The obvious way to prove the committed fixtures still match live R is to
regenerate them and run `git diff --exit-code`. That is wrong, and it took a
CI run to make the reason concrete.

Byte identity of 17-significant-digit doubles additionally requires both
machines to agree on floating-point *accumulation*. R sums in `LDOUBLE`, so a
build with `capabilities("long.double") == TRUE` — Linux x86_64, which is what
the CI container is — and one without — the conda-forge macOS arm64 build the
fixtures were generated on — differ in the last ulp of every group mean.
Neither is wrong. This document already said the gold standard is a *specific*
R build; the byte gate quietly assumed otherwise.

So CI compares the two fixture sets under the contract the project actually
makes: **structure exactly, numbers to 1e-6**
(`parity/compare_fixtures.py`). That is stronger than byte equality where it
counts — a changed column, a lost factor level, a reordered row or a genuinely
shifted value all fail — and it does not fail for a last digit no diffusion
tube could resolve. A byte-level `git diff --stat` still runs alongside it, as
information rather than a gate.

Two things that comparison has to do for itself, because the loaded pandas
Series cannot carry them:

- **`NA_real_` vs `NaN`.** `load_column` maps both to `float64` nan, since that
  is all pandas has. The distinction survives only in the serialised token, so
  the tokens are compared as text.
- **Factor levels and their order, character NA positions, POSIXct `tzone`.**
  These sit beside `values` in the node and are compared exactly.

`_lock.json` and `_shims.json` are excluded: they record R package versions and
which shims were needed, which legitimately differ per machine. The parts of
them that must not drift — the upstream SHA and the R-source hash — are gated
by `tests/test_lock.py` instead.

## Pinned environment

The gold standard is a *specific* R build, not "R in general". Several results
below depend on how R's C sources were compiled.

- Upstream DTEval: `08a6cbda6d5703ef564f076f5ae5aafc1d7ad6d6` (2026-06-30)
- R 4.6.1, `aarch64-apple-darwin20`, conda-forge build
- `capabilities("long.double")` is `FALSE` — `LDOUBLE` is 64-bit, so R's
  accumulations are plain double arithmetic
- Locale pinned to `LC_COLLATE=C`, `LC_TIME=C`, `LC_NUMERIC=C` and a UTF-8
  `LC_CTYPE` (see below)

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

`OpenStreetMap` is a different kind of shim, and the distinction matters. It
is reached only by `tubeMap`, to fetch ESRI basemap tiles through rJava. The
port does not fetch tiles at all — it hands the frontend a basemap *request* —
so there is nothing to compare, and no upstream source worth re-emitting. What
`tools/build_r_shims.R` installs is a **stand-in**, not a port: `openmap`
returns the requested bounding box with an empty tile list, and `openproj` is
the identity. Nothing in it comes from upstream, and `_shims.json` says so.

The point is not to reproduce the basemap but to make everything *around* it
reproducible: without a stand-in, `tubeMap` cannot run in R here at all, and
its extent arithmetic, layer structure and limits would go untested. With one,
they are gated like anything else — and the raster stays out of scope, which it
would be either way. Read the `map/tubeMap.*` cases with that in mind: they
prove the plot is assembled as R assembles it, not that the tiles match.

### The reference-monitor data set

`testTubeAccuracy` compares tubes against a co-located continuous analyser, and
upstream's examples fetch that with `openair::importAURN` — a network call,
which cannot be a fixture. `tools/aurn_example.R` builds a deterministic
openair-shaped stand-in instead: hourly NO₂ for 2022–2026 at two sites, one
placed exactly on a real `dt.brd` tube location so the distance test has a
match, with a seasonal and a diurnal cycle plus seeded noise.

Both sides read the same generator: `tools/export_datasets.R` ships it to
Python as `dteval.datasets.aurn_example()`, and `parity/generate.R` sources it
to put `aurn.example` in scope for the R expressions. The two are identical by
construction rather than by a file round-trip. The generator saves and restores
`.Random.seed`, so seeding it does not perturb the RNG stream `clusterTubeData`
draws from later in the same run.

### AQEval on the Python side

The Python port depends on the user's own
[aqeval_python](https://github.com/chris-r-uol/aqeval_python) rather than
reimplementing `find_near_lat_lon` / `calc_date_range_stat`.

Its Haversine originally agreed with R to ~1e-10 m but not bit-for-bit: over
1,884 real site pairs from `dt.brd`, 1868 distances differed. Two
floating-point details, not a difference in formula — R converts the coordinate
*difference* from degrees, and associates the second term as
`sin(dLon/2) * sin(dLon/2) * cos(lat0) * cos(lat1)`. Corrected upstream (branch
`fix/haversine-r-parity`), all 1,884 now match exactly.

At the current 1e-6 bound the difference is invisible anyway (4e-14 relative),
so `tubeSummaryLatLon` no longer needs a tolerance for it — but the fix is
still worth having, since it costs nothing and removes a needless divergence.

## Findings that shaped the port

These are not incidental; each one silently changes output values if ignored.

### `.sample_id` — grouping reproduced, numbering not

R builds the key by pasting latitude, longitude and the two dates into a
string and taking `as.numeric(factor(...))`. Its integers therefore depend on
how R formats a double (15 significant digits, not Python's shortest
round-trip) *and* on collation order.

We group on the `(lat, lon, start, end)` **tuple** directly — no string round
trip, so nothing depends on float formatting. The replicate sets are identical;
the integer labels are not necessarily the same.

The parity suite checks this properly rather than ignoring the column:
`label_columns` verifies the induced **partition** matches R's as a bijection —
two rows share a label here exactly when they share one in R — so a genuine
mis-grouping still fails.

This replaced a ~480-line port of R's `format.c`. `rcompat/numfmt.py` now holds
only `signif` (report strings embed `signif(x, 4)`) and a simple
`as_character` (`.location` renders as `"{lat,lon}"`).

Two consequences, both handled by canonicalising row order before comparing
(`row_order_artifact` in the manifest):

- `testTubePrecision`'s row order depends on `.sample_id`, because R sorts its
  merge key on the label as text.
- `testTubeMeta(plot.type = 2)` keeps one stacked bar segment per group rather
  than aggregating, and the per-sample segments come out in `.sample_id` order.
  Segment order within a stacked bar carries no meaning; all 91,110 rows match
  exactly on `variable`, `ref`, `value` and `..type`.

### `LC_CTYPE` decides what the data *says*

`LC_COLLATE`, `LC_TIME` and `LC_NUMERIC` were pinned from the start.
`LC_CTYPE` was not, and it turns out to change fixture content rather than
formatting — which only surfaced when CI regenerated on Linux and 42 fixtures
disagreed.

`dt.brd`'s `site_name` contains non-ASCII: real typographic quotes in
*Outside ‘Vapes and Phones’ shop*, and a stray `0x96` (a cp1252 en-dash that
was never re-encoded upstream) in *Low Mill, Keighley*. Under a C `LC_CTYPE`,
`load()` translates those strings to UTF-8 and serialisation then escapes them
as the literal seven characters `<U+2018>`. Under a UTF-8 `LC_CTYPE` the
characters survive as themselves. Both are legitimate R behaviour; only one can
be the fixture, and leaving it to the ambient environment meant the answer
depended on whose shell ran the generator.

`parity/ctype.R` now pins it to the first available UTF-8 locale (`C.UTF-8` on
both a current macOS R and the CI container), and is sourced by
`parity/generate.R`, `parity/generate_rcompat.R` and `tools/export_datasets.R`
— all three, because the shipped Python datasets are converted by the same
mechanism and have to agree with the fixtures.

UTF-8 is the pinned choice because the quotes are real characters in the source
and a frontend rendering `<U+2018>` to a user would be plainly wrong. Note the
consequence for the `0x96`: it is preserved as U+0096, a C1 control character.
That is faithful to the source data rather than tidy, which is the right way
round for a port — `dteval.datasets.dt_brd()` reproduces what upstream has,
including its encoding damage.

### What is deliberately *not* recorded

`parity/fixtures/_rcompat.json.gz` used to carry a `numfmt` block: `sprintf("%a")`
bit patterns for ~5,500 doubles, the reference for the ~480-line `format.c`
port that went when parity moved to 1e-6. Nothing had read it since, and it
could never be portable — `sprintf("%a")` is the C library's, and glibc renders
the smallest denormal as `0x0.0000000000001p-1022` where macOS writes
`0x1p-1074`. It produced ~21,700 spurious differences on regeneration,
drowning the two real ones. Removed.

### Collation order is a value, not a presentation detail

`factor()` levels are `sort(unique(x))` under the current collation, and level
order decides how grouped results are ordered. Row order of
`testTubePrecision` and `testTubeAccuracy` output comes from
`sort(unique(data$.cut))`.

The port reproduces **R under `LC_COLLATE=C`**, which is pinned in the fixture
generator. A user running R under a UK/US locale — where collation largely
ignores punctuation at the primary strength — can legitimately get a different
ordering. Supporting full ICU collation is possible but would add a heavy
dependency; raise an issue if you need it.

(`.sample_id` no longer depends on this: it is built from a tuple, not a
collated string.)

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

### R's floating-point arithmetic — found, then deliberately dropped

Under the old bit-exact bar two behaviours had to be reproduced, and both are
worth recording even though the code is gone:

- **`mean` is two-pass.** R computes `sum(x)/n`, then corrects it with
  `s += sum(x - s)/n`. Skipping the correction changes the last bit of nearly
  every group mean.
- **R's compiler contracts multiply-adds into FMAs.** Its variance loop is
  `sum += (x[k] - xm) * (x[k] - xm)`, fused under the default
  `-ffp-contract=on`. Reproducing that matched **2875/2875** real replicate
  groups where plain sequential accumulation matched 2262.

Both are now handled by numpy, agreeing with R to ~1e-16 — seven orders inside
the 1e-6 bound. The ~520 lines that chased them (including hand-ported `qnorm`
and `qt`) were removed, along with their dependence on `long double` being
64-bit.

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

### LOESS — three different situations

R's `stats::loess` wraps netlib's `dloess`. What the port can guarantee depends
on the number of predictors and the surface.

| case | used by | status |
|---|---|---|
| 1 predictor, either surface | `testTubePrecision` | `scikit-misc` (same `dloess`), ~10 ulps |
| ≥2 predictors, `surface="direct"` | `fitTubeModel_loess` | native, **2.9e-13** relative |
| ≥2 predictors, `surface="interpolate"` | `deseasonTubeData` | **not available** — raises |

**`scikit-misc` 0.5.2's multivariate LOESS does not fit correctly.** With
`y = u` exactly (noiseless) and an irrelevant second predictor `v`: one
predictor recovers `y` to 1e-13; two predictors return a near-constant (fitted
sd 5.6 against `y`'s 30.1, worst error 60 over a range of 100), for both
surfaces. Against R on a smooth surface the relative error reaches 2.6. So
`r_loess` **raises** for the multivariate interpolating case rather than
returning plausible-looking wrong numbers, which would corrupt
`deseasonTubeData` while looking fine.

`surface="direct"` is implemented natively in `dteval._loess_direct` — plain
tricube-weighted local polynomial regression, no kd-tree — and reproduces R's
fitted values to a few ulps.

#### `se.fit` is the weak spot

R scales the standard error by a residual scale derived from an *approximate*
`delta1` (`statistics = "1.approx"`), computed by the Fortran `lowesa` →
`ehg141`, which evaluates a hard-coded spline via `ehg128` — 339 lines of
tensor-product blending. We use the exact hat matrix instead, landing **2.7e-3
relative** from R.

At that bound, `.value.pred.se` is a *structural* check rather than a numeric
one. The fitted values are unaffected. Porting `ehg141`/`ehg176`/`ehg128` would
fix it — and since `ehg128` is the same kd-tree interpolator the
`surface="interpolate"` path needs, it would unblock `deseasonTubeData` too.
That is the single highest-value remaining piece of numerical work.

### `deseason_tube_data` — R's LOESS surface, not ours

> Regenerating the fixtures on Linux put a number on how unstable R's own
> approximation is: `..season` moves by up to **8e-6 relative** between R on
> macOS arm64 and R on Linux x86_64 — above the 1e-6 parity tolerance, from the
> same R source on the same data. R's kd-tree surface is not merely an
> approximation of the exact fit, it is not reproducible across platforms
> either. Everything else in that fixture agrees to 1e-6.


Structure matches exactly. The fitted components (`..fit`, `..trend`,
`..season`, `..deseason`) do not, and this is the one place the port
deliberately computes something different from R's default.

R fits with LOESS's default `surface="interpolate"`, a kd-tree approximation of
the local regression built for speed. We fit `surface="direct"`, the exact
local regression, because `scikit-misc`'s multivariate path is broken and
reproducing R's kd-tree means porting `ehg128`.

**The gap is R's approximation error, not ours.** `tests/test_loess.py` pins
both halves: our fit matches R's *own* `surface="direct"` to **2e-13**, and R's
two surfaces differ from each other by up to **2.9 µg/m³** on a 17-point group.

Against R's default output, over all 157 locations:

| | µg/m³ |
|---|---|
| median | 0.15 |
| p95 | 1.2 |
| p99 | 3.6 |
| max | 11.7 |

The median sits well inside the ~10% (~2.4 µg/m³) measurement error; the tail
does not. It is worst for small groups, where R's kd-tree cells are coarse
relative to the data and R itself warns of near singularities. Marked
non-gating and stated plainly, rather than hidden under a tolerance that would
have to be ~100% to pass.

### `cluster_tube_data` — PAM, not clara

Everything up to the clustering matches: the site-by-site `1 - correlation`
feature matrix agrees with R to **3e-15**, and the result's columns, column
order and row count are identical. The cluster assignment differs.

`clusterTubeData` calls `cluster::clara` — PAM run on random *subsamples* for
speed, 44 of the 157 sites by default — so its answer depends on R's RNG
stream. The port runs PAM on the full data: deterministic, and the exact
solution clara approximates.

Again, not the port being worse. From `tests/test_cluster.py`:

- our PAM reproduces **R's own `pam()`** partition for all 157 sites;
- R's `clara` disagrees with R's own `pam()` on **100** of them;
- and scores a worse objective — 2.1083 against 2.0964.

The divergence is clara-vs-PAM *inside R*. Reproducing it would mean porting
R's Mersenne-Twister and clara's sampling scheme in order to inherit a worse
answer.

### `fit_tube_model_gam` — the optimiser, not the model

`fitTubeModel_gam` fits `[tube] ~ te(lon, lat)` with `mgcv::gam`. `dteval.gam`
reproduces mgcv's *construction* exactly — cubic-regression-spline marginals
with `k = 5`, knots at quantiles of the **unique** covariate values, a
tensor-product basis, one wiggliness penalty per marginal direction, a
sum-to-zero constraint, and GCV — on numpy and scipy alone.

What it does not reproduce is mgcv's smoothing-parameter *optimiser*, a nested
Newton scheme with its own reparameterisations and step control. The GCV
objective here is very flat near its minimum: **44.30** at our 24.6 effective
degrees of freedom against **44.64** at 19.5. R settles at 22.8 edf, inside
that same flat region. Two defensible answers to the same criterion.

Measured over all 11,273 rows of `dt.brd`, our fitted surface against R's:

| | |
|---|---|
| median absolute difference | 0.033 µg/m³ |
| p95 | 0.17 µg/m³ |
| max | **1.32 µg/m³** |
| correlation | 0.99967 |

The worst case sits inside the ~2.4 µg/m³ (10%) measurement error on a
diffusion tube. The gap is confined to `.value.pred` and `.value.pred.se`:
every other column matches R exactly, which `tests/test_gam.py` asserts, along
with the basis properties (the `cr` basis is the identity at its knots, the
penalty annihilates straight lines, a noiseless tensor surface is recovered to
1e-6) and the deviation bounds above.

The knot placement was worth finding. mgcv uses quantiles of the *unique*
values; using quantiles of the raw column instead bunches knots around the
busiest sites — tube coordinates repeat once per sampling period — and pushed
the max deviation from 1.32 to 1.73.

### `tube_map` — everything but the tiles

R annotates an `OpenStreetMap` ESRI raster under the plot layers. The port
issues a basemap *request* instead (`spec["basemap"]`: provider, projection and
bounding box) and leaves the drawing to the client, which is what a Svelte or
maplibre frontend wants anyway. Fetching different tiles from a different
provider would not make the figure more faithful.

Everything else is gated: the `grid.borders` extent arithmetic, the layer
structure (including that R calls `tubePlot` **twice**, so the tube layers
appear twice, with `expand_limits`' `geom_blank` between them), the cleared
axis labels, the zeroed scale expansion and `coord_quickmap`. See the shim note
above for how the R side is made runnable.

### `leaflet_tube_map` — compared on layer content

leaflet and folium emit different HTML, so the comparison is on what DTEval
decides: the ordered list of layer calls, and per call the coordinates, radii
and colours. `parity/serialize.R` names leaflet's positional argument lists per
method and keeps only those; `compare_leaflet` in `parity/compare.py` checks
them. The `map/leafletTubeMap.coloured` case compares 11,273 marker positions
and 11,273 colours exactly.

Two details were needed for that:

- **Colours interpolate in CIE Lab.** `leaflet::colorNumeric` goes through
  `scales::colour_ramp`, which converts the palette to Lab, interpolates there
  and converts back. Interpolating in sRGB instead is off by one or two levels
  per channel — `#F57547` where R gives `#F67647`.
- **`pretty()` on a flat range differs, deliberately.** The legend breaks use
  R's `pretty()`, reproduced for any range with width. A *zero-width* range
  takes a separate branch in R's C code that invents a span around the value
  (`pretty(c(5, 5))` is `0 5`); the port returns the single value. It only
  arises for a legend on a perfectly flat fitted surface, where one break is
  the more useful answer.

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

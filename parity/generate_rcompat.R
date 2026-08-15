#!/usr/bin/env Rscript
#############################################################
# parity/generate_rcompat.R
#############################################################
#
# Reference values for the R-semantics primitives in dteval.rcompat.
#
# These are generated separately from parity/generate.R because they test R
# itself (as.character, mean, sd, quantile, qt, factor levels, Date formatting)
# rather than DTEval. Everything in the port is built on them, so they are
# verified first and independently -- a bug here would otherwise show up as a
# baffling failure three layers up.
#
# Usage:  Rscript parity/generate_rcompat.R

invisible(Sys.setlocale("LC_COLLATE", "C"))
invisible(Sys.setlocale("LC_TIME", "C"))
invisible(Sys.setlocale("LC_NUMERIC", "C"))
suppressPackageStartupMessages(library(jsonlite))

root <- getwd()
out_dir <- file.path(root, "parity", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
R_REF <- Sys.getenv("DTEVAL_R_REFERENCE", file.path(root, "r_reference"))

set.seed(1)
env <- new.env(); load(file.path(R_REF, "data", "dt.brd.rda"), envir = env)
d <- env$dt.brd

o <- list()

# ---- number formatting ----------------------------------------------------
vals <- c(
  as.numeric(d$latitude), as.numeric(d$longitude),
  as.numeric(d$bias_adjusted_measurement),
  as.numeric(d$.start_date), as.numeric(d$.end_date),
  c(0, -0, 1, -1, 0.5, 2.5, 1/3, pi, exp(1), 1e-20, 1e20, 1e5, 1e15, 1e16,
    1e-5, 1e-4, 0.1, 0.2, 0.1 + 0.2, 100000.5, 999999.5, 1e100, 1e-100,
    1e308, 5e-324, 123456789012345678, 1.000000000000001, -1e-300),
  runif(2000, -1e3, 1e3),
  rnorm(2000) * 10^sample(-12:12, 2000, TRUE),
  round(runif(1000, 0, 200), 4), (1:500)/7, 10^(-30:30)
)
vals <- unique(vals[is.finite(vals)])
o$numfmt <- list(
  bits = sprintf("%a", vals),
  as_character = as.character(vals),
  signif4 = sprintf("%a", signif(vals, 4)),
  signif1 = sprintf("%a", signif(vals, 1)),
  signif15 = sprintf("%a", signif(vals, 15))
)

# ---- Date formatting, including fractional dates --------------------------
dd <- c(0, 1, 19000, 19000.5, 18999.5, -0.5, -1.5, 19098.5, 19099.5,
        100.25, 100.75, 12345.5, -1000.75)
o$dates <- list(
  days = sprintf("%.17g", dd),
  ymd = format(as.Date(dd, origin = "1970-01-01"), "%Y-%m-%d"),
  j = format(as.Date(dd, origin = "1970-01-01"), "%j"),
  b = format(as.Date(dd, origin = "1970-01-01"), "%b"),
  B = format(as.Date(dd, origin = "1970-01-01"), "%B")
)

# month-end via the POSIXlt increment trick used by tagTubeStartEnd method 1
yy <- rep(2015:2026, each = 12); mm <- rep(1:12, 12)
t0 <- as.POSIXlt(paste("01", month.abb[mm], yy, sep = "/"), format = "%d/%b/%Y")
st <- as.Date(t0); t0$mon <- t0$mon + 1; en <- as.Date(t0) - 1
o$month_bounds <- list(year = yy, month = mm,
                       start = sprintf("%.17g", as.numeric(st)),
                       end = sprintf("%.17g", as.numeric(en)))

# ---- collation and factor levels ------------------------------------------
keys <- unique(c(
  apply(d[c("latitude", "longitude", ".start_date", ".end_date")], 1,
        paste, collapse = "-"),
  c("a", "B", "_z", ".x", "10", "2", "Z-1", "z-1", "", " a", "A", "~", "!")
))
o$collate <- list(input = keys, sorted = sort(keys),
                  levels = levels(factor(keys)),
                  codes = as.numeric(factor(keys)))

# ---- aggregation on DTEval's real domain ----------------------------------
# Co-located replicate groups: exactly what testTubePrecision operates on.
key <- paste(d$latitude, d$longitude, d$.start_date, d$.end_date)
sp <- split(d$bias_adjusted_measurement, key)
sp <- sp[vapply(sp, length, 1L) >= 2]
o$agg_real <- list(
  x = unname(lapply(sp, function(v) sprintf("%.17g", v))),
  mean = unname(vapply(sp, function(v) sprintf("%.17g", mean(v, na.rm = TRUE)), "")),
  sd = unname(vapply(sp, function(v) sprintf("%.17g", sd(v, na.rm = TRUE)), "")),
  var = unname(vapply(sp, function(v) sprintf("%.17g", var(v, na.rm = TRUE)), "")),
  median = unname(vapply(sp, function(v) sprintf("%.17g", median(v, na.rm = TRUE)), "")),
  q = unname(lapply(sp, function(v)
    sprintf("%.17g", quantile(v, c(0, 0.025, 0.25, 0.5, 0.75, 0.975, 1),
                              na.rm = TRUE, names = FALSE))))
)

# ---- adversarial aggregation domain ---------------------------------------
# Wide magnitude range and larger n. Recorded so the port's known 1-ulp
# var/sd ambiguity outside DTEval's domain stays measured rather than assumed.
set.seed(11)
sets <- lapply(1:400, function(i) {
  n <- sample(2:60, 1); rnorm(n) * 10^sample(-6:6, 1) + runif(1, -50, 50)
})
o$agg_synthetic <- list(
  x = lapply(sets, function(s) sprintf("%.17g", s)),
  mean = vapply(sets, function(s) sprintf("%.17g", mean(s)), ""),
  sd = vapply(sets, function(s) sprintf("%.17g", sd(s)), ""),
  var = vapply(sets, function(s) sprintf("%.17g", var(s)), "")
)

# ---- qt / qnorm -----------------------------------------------------------
dfs <- c(1, 1.5, 2, 2.0000000001, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100,
         200, 1000, 1e21)
ps <- c(1e-8, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
        0.975, 0.99, 0.999, 1 - 1e-8)
o$qt <- list(df = dfs,
             upper = vapply(dfs, function(v) sprintf("%.17g", qt(0.05/2, df = v, lower.tail = FALSE)), ""),
             lower = vapply(dfs, function(v) sprintf("%.17g", qt(0.05/2, df = v, lower.tail = TRUE)), ""))
o$qnorm <- list(p = ps, v = vapply(ps, function(v) sprintf("%.17g", qnorm(v)), ""))

# ---- cut labels and summary(factor) ---------------------------------------
br <- c(0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000)
vv <- c(0.5, 5, 50, 500, 5e4, 5e6, 0, NA, 1e8)
cc <- cut(vv, breaks = br)
o$cut <- list(levels = levels(cc), codes = as.integer(cc))
f <- factor(c(rep(1, 4107), rep(2, 410), rep(3, 3111)))
sf <- data.frame(t(summary(f)))
o$summary_factor <- list(names = names(sf), values = as.integer(sf[1, ]))

# gzipped for the same reason as the case fixtures: large, highly repetitive,
# and gzfile() is deterministic so the CI staleness diff still works.
out_path <- file.path(out_dir, "_rcompat.json.gz")
con <- gzfile(out_path, "wt")
writeLines(jsonlite::toJSON(o, digits = NA, na = "string", auto_unbox = FALSE), con)
close(con)
message("wrote ", out_path)

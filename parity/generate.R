#!/usr/bin/env Rscript
#############################################################
# parity/generate.R
#############################################################
#
# Runs every case in parity/manifest.yaml against the upstream DTEval R package
# and writes parity/fixtures/<id>.json, plus parity/fixtures/_lock.json
# recording exactly which R sources produced them.
#
# Usage:  Rscript parity/generate.R [case-id-substring ...]
#
# The locale is pinned (see manifest meta.locale). R's sort(), factor() level
# order and strptime month parsing are all locale-sensitive, and DTEval's
# .sample_id values depend on collation order -- so the gold standard has to be
# generated under a stated locale, not whatever the machine happens to use.

suppressPackageStartupMessages({
  library(jsonlite)
})

invisible(Sys.setlocale("LC_COLLATE", "C"))
invisible(Sys.setlocale("LC_TIME", "C"))
invisible(Sys.setlocale("LC_NUMERIC", "C"))
options(stringsAsFactors = FALSE, warn = 1)

`%||%` <- function(a, b) if (is.null(a)) b else a

# Run from the repo root; fall back to the working directory when sourced.
root <- getwd()
if (!dir.exists(file.path(root, "parity"))) {
  stop("run parity/generate.R from the repository root", call. = FALSE)
}

source(file.path(root, "parity", "serialize.R"))

R_REF <- Sys.getenv("DTEVAL_R_REFERENCE", file.path(root, "r_reference"))
if (!dir.exists(R_REF)) {
  stop("Upstream R package not found at '", R_REF, "'.\n",
       "  Run: make r-reference   (clones the pinned SHA)\n",
       "  or set DTEVAL_R_REFERENCE=/path/to/DTEval", call. = FALSE)
}

# ---------------------------------------------------------------- manifest --
read_manifest <- function(path) {
  if (requireNamespace("yaml", quietly = TRUE)) {
    return(yaml::read_yaml(path))
  }
  stop("The 'yaml' R package is required to read the parity manifest.\n",
       "  install.packages('yaml')", call. = FALSE)
}

manifest <- read_manifest(file.path(root, "parity", "manifest.yaml"))
cases <- manifest$cases

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  keep <- vapply(cases, function(cs) any(vapply(args, function(a) grepl(a, cs$id, fixed = TRUE), logical(1))), logical(1))
  cases <- cases[keep]
  message("filtered to ", length(cases), " case(s)")
}

# ------------------------------------------------------------ load DTEval --
# Install the pinned tree into a project-local library and load it normally,
# rather than sourcing R/*.R. The package uses `DTEval::dt.calendar` internally
# (tagTubeStartEnd method 2), and `::` resolves against a real namespace -- a
# sourced environment cannot stand in for one. Installing from r_reference/ at
# the pinned SHA keeps the provenance guarantee either way.
lib <- file.path(root, ".Rlib")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

# The package reaches AQEval and OpenStreetMap only through `::` -- neither
# appears in its NAMESPACE -- so they are install-time dependency checks, not
# load-time imports. OpenStreetMap needs rJava and therefore a JRE, which is
# not always available. We install from a copy of the pinned tree with those
# two relaxed out of Imports: behaviour-neutral for every function that does
# not call them, and the ones that DO call them fail loudly at run time and are
# reported as failed cases rather than silently skipped.
#
# The relaxation is recorded in _lock.json so it is visible in the fixture
# provenance.
# AQEval is normally supplied by tools/build_r_shims.R (see that file for why).
# OpenStreetMap needs rJava and is only used for tubeMap basemaps, which are
# not parity-tested, so it stays relaxed when unavailable.
RELAXED_IMPORTS <- c("AQEval", "OpenStreetMap")
if (!requireNamespace("AQEval", quietly = TRUE)) {
  message("note: AQEval not installed. Run `make r-shims` to build the minimal\n",
          "      shim, otherwise testTubeAccuracy / deseason(method=2) /\n",
          "      tubeSummaryLatLon fixtures cannot be generated.")
}

build_dir <- file.path(tempdir(), "DTEval-build")
unlink(build_dir, recursive = TRUE)
dir.create(dirname(build_dir), recursive = TRUE, showWarnings = FALSE)
file.copy(R_REF, dirname(build_dir), recursive = TRUE)
file.rename(file.path(dirname(build_dir), basename(R_REF)), build_dir)

desc_path <- file.path(build_dir, "DESCRIPTION")
desc <- readLines(desc_path, warn = FALSE)
desc_txt <- paste(desc, collapse = "\n")
relaxed <- character(0)
for (pkg in RELAXED_IMPORTS) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    desc_txt <- gsub(paste0(",\\s*", pkg, "\\s*(\\([^)]*\\))?"), "", desc_txt)
    desc_txt <- gsub(paste0(pkg, "\\s*(\\([^)]*\\))?,\\s*"), "", desc_txt)
    relaxed <- c(relaxed, pkg)
  }
}
writeLines(strsplit(desc_txt, "\n")[[1]], desc_path)
if (length(relaxed)) {
  message("relaxed unavailable Imports for install: ", paste(relaxed, collapse = ", "))
}

message("installing DTEval from ", R_REF, " into ", lib)
inst <- suppressWarnings(system2("R", c("CMD", "INSTALL", "--no-docs",
                                        paste0("--library=", shQuote(lib)),
                                        shQuote(build_dir)),
                                 stdout = TRUE, stderr = TRUE))
if (!is.null(attr(inst, "status")) && attr(inst, "status") != 0) {
  stop("R CMD INSTALL failed:\n", paste(inst, collapse = "\n"), call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(methods); library(DTEval)
})

# datasets are lazy-loaded by the package; bring them into scope by name
data_env <- new.env()
for (nm in c("dt.brd", "dt.calendar", "caz.brd")) {
  assign(nm, get(nm, envir = asNamespace("DTEval")), envir = data_env)
}
attach(data_env, name = "DTEval_data", warn.conflicts = FALSE)

# ------------------------------------------------------------------- run ---
fixture_dir <- file.path(root, "parity", "fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

ok <- 0L; failed <- character(0)
for (cs in cases) {
  id <- cs$id
  message("[case] ", id)
  val <- tryCatch(
    eval(parse(text = cs$r), envir = new.env(parent = globalenv())),
    error = function(e) structure(conditionMessage(e), class = "dte_case_error")
  )
  if (inherits(val, "dte_case_error")) {
    message("   !! ", val)
    failed <- c(failed, id)
    next
  }
  if (!is.null(cs$compare)) {
    for (part in strsplit(cs$compare, ".", fixed = TRUE)[[1]]) val <- val[[part]]
  }
  write_fixture(id, val, dir = fixture_dir,
                meta = list(r_expr = cs$r, compare = cs$compare %||% NA_character_))
  ok <- ok + 1L
}

# ------------------------------------------------------------------ lock ---
# Records what produced these fixtures. tests/test_lock.py fails if the pinned
# upstream tree no longer hashes to this, which is what stops fixtures from
# silently going stale.
hash_tree <- function(dir) {
  fs <- sort(list.files(dir, recursive = TRUE, full.names = TRUE))
  fs <- fs[!grepl("(^|/)[.]git(/|$)", fs)]
  rel <- substring(fs, nchar(dir) + 2L)
  paste0(
    substr(digest_or_tools(paste(rel, vapply(fs, tools::md5sum, character(1)), collapse = "\n")), 1, 64)
  )
}
digest_or_tools <- function(s) {
  tf <- tempfile(); writeLines(s, tf); on.exit(unlink(tf))
  unname(tools::md5sum(tf))
}

pkgs <- c("data.table", "ggplot2", "AQEval", "mgcv", "cluster", "sf")
pkg_versions <- setNames(
  lapply(pkgs, function(p) {
    v <- tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
    v
  }), pkgs)

lock <- list(
  ser_version = DTE_SER_VERSION,
  generated_by = "parity/generate.R",
  upstream_repo = manifest$meta$upstream_repo,
  upstream_sha = manifest$meta$upstream_sha,
  r_source_hash = hash_tree(file.path(R_REF, "R")),
  r_data_hash = hash_tree(file.path(R_REF, "data")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  locale = manifest$meta$locale,
  relaxed_imports = if (length(relaxed)) relaxed else character(0),
  shims = if (file.exists(file.path(fixture_dir, "_shims.json"))) {
    jsonlite::fromJSON(file.path(fixture_dir, "_shims.json"))
  } else NULL,
  packages = pkg_versions,
  n_cases = ok
)
writeLines(jsonlite::toJSON(lock, auto_unbox = TRUE, pretty = TRUE),
           file.path(fixture_dir, "_lock.json"))

message("\nwrote ", ok, " fixture(s) to ", fixture_dir)
if (length(failed)) {
  message("FAILED cases (", length(failed), "):")
  for (f in failed) message("   ", f)
  quit(status = 1L)
}

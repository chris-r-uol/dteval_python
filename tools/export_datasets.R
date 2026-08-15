#!/usr/bin/env Rscript
#############################################################
# tools/export_datasets.R
#############################################################
#
# Converts the upstream .rda datasets into the typed-JSON form the Python
# package ships and reads (src/dteval/data/).
#
# Using the same lossless format as the parity fixtures -- rather than CSV --
# means dt.brd arrives in Python with the classes it had in R: integer stays
# integer, Date stays Date, factor levels keep their order. A CSV round-trip
# would guess, and a wrong guess propagates into every downstream result.
#
# The exported data is itself parity-checked (cases data/dt.brd,
# data/dt.calendar), so a conversion error cannot pass silently.
#
# Usage:  Rscript tools/export_datasets.R

invisible(Sys.setlocale("LC_COLLATE", "C"))
invisible(Sys.setlocale("LC_TIME", "C"))
suppressPackageStartupMessages(library(jsonlite))

root <- getwd()
if (!dir.exists(file.path(root, "parity"))) {
  stop("run tools/export_datasets.R from the repository root", call. = FALSE)
}
source(file.path(root, "parity", "serialize.R"))

R_REF <- Sys.getenv("DTEVAL_R_REFERENCE", file.path(root, "r_reference"))
out_dir <- file.path(root, "src", "dteval", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

env <- new.env()
for (rda in list.files(file.path(R_REF, "data"), pattern = "[.]rda$", full.names = TRUE)) {
  load(rda, envir = env)
}

write_gz <- function(obj, name) {
  path <- file.path(out_dir, paste0(name, ".json.gz"))
  txt <- jsonlite::toJSON(
    list(name = name, value = serialize_value(obj)),
    auto_unbox = TRUE, null = "null", na = "string", digits = NA
  )
  con <- gzfile(path, "wt"); writeLines(txt, con); close(con)
  message("wrote ", path, " (", format(file.size(path) / 1024, digits = 4), " KiB)")
}

# --- plain data frames -----------------------------------------------------
write_gz(env$dt.brd, "dt_brd")
write_gz(env$dt.calendar, "dt_calendar")

# --- caz.brd is an sf object: export geometry as GeoJSON -------------------
# sf/shapely both speak GeoJSON, and the CAZ boundary is only ever used for
# point-in-polygon tests, so the geometry is the whole payload.
suppressPackageStartupMessages(library(sf))
caz <- sf::st_transform(env$caz.brd, crs = "WGS84")
geo_path <- file.path(out_dir, "caz_brd.geojson")
if (file.exists(geo_path)) file.remove(geo_path)
sf::st_write(caz, geo_path, driver = "GeoJSON", quiet = TRUE,
             layer_options = c("COORDINATE_PRECISION=15"))
message("wrote ", geo_path)

message("\ndone.")

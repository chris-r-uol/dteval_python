#############################################################
# tools/aurn_example.R
#############################################################
#
# The synthetic reference (continuous analyser) data set.
#
# testTubeAccuracy compares diffusion tubes against a co-located reference
# monitor, and the R examples fetch that from openair::importAURN over the
# network -- which cannot be a fixture. This builds a deterministic
# openair-shaped stand-in instead: hourly NO2 at two sites, one placed exactly
# on a real dt.brd tube location so the distance test has something to match.
#
# Sourced by BOTH tools/export_datasets.R (which ships it to Python) and
# parity/generate.R (which needs it in scope to evaluate the R side of the
# accuracy cases). Same seed, same code, so the two are identical by
# construction rather than by a file round-trip.

make_aurn_example <- function(dt.brd) {
  # leave the global RNG as we found it -- generate.R calls this before the
  # cases run, and clusterTubeData's clara draws from that same stream
  if (exists(".Random.seed", envir = globalenv())) {
    old <- get(".Random.seed", envir = globalenv())
    on.exit(assign(".Random.seed", old, envir = globalenv()))
  } else {
    on.exit(suppressWarnings(rm(".Random.seed", envir = globalenv())))
  }
  set.seed(20260815)
  tubes <- unique(dt.brd[c("latitude", "longitude")])
  tubes <- tubes[!is.na(tubes$latitude), ]
  anchor <- tubes[1, ]
  hours <- seq(as.POSIXct("2022-01-01 00:00:00", tz = "UTC"),
               as.POSIXct("2026-01-31 23:00:00", tz = "UTC"), by = "hour")
  doy <- as.numeric(format(hours, "%j"))
  hr <- as.numeric(format(hours, "%H"))
  mk <- function(lat, lon, site, code, base) {
    no2 <- base +
      8 * cos(2 * pi * (doy - 15) / 365) +      # winter maximum
      6 * sin(2 * pi * (hr - 8) / 24) +          # diurnal cycle
      rnorm(length(hours), 0, 4)
    data.frame(date = hours, no2 = pmax(no2, 0.5), site = site, code = code,
               source = "synthetic", latitude = lat, longitude = lon)
  }
  rbind(
    mk(anchor$latitude, anchor$longitude, "Bradford Synthetic A", "SYNA", 28),
    mk(anchor$latitude + 0.02, anchor$longitude + 0.02, "Bradford Synthetic B", "SYNB", 20)
  )
}

#!/usr/bin/env Rscript
# Render every catalogue figure with R DTEval, and dump each plot's layer data
# so the Python side can be checked against it rather than eyeballed.
invisible(Sys.setlocale("LC_COLLATE", "C"))
invisible(Sys.setlocale("LC_TIME", "C"))
invisible(Sys.setlocale("LC_NUMERIC", "C"))

root <- Sys.getenv("DTEVAL_ROOT", getwd())
out  <- Sys.getenv("FIG_OUT", file.path(root, "build", "figures", "r"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
source(file.path(root, "parity", "ctype.R")); dte_pin_ctype()
source(file.path(root, "parity", "serialize.R"))

suppressPackageStartupMessages({
  library(yaml); library(jsonlite); library(ggplot2); library(DTEval)
})

dt.brd <- get("dt.brd", envir = asNamespace("DTEval"))
source(file.path(root, "tools", "aurn_example.R"))
aurn.example <- make_aurn_example(dt.brd)

cat <- yaml::read_yaml(file.path(root, "tools", "figures", "catalogue.yaml"))
specs <- list()

for (fig in cat$figures) {
  message("[R] ", fig$id)
  p <- eval(parse(text = fig$r), envir = environment())
  ggplot2::ggsave(file.path(out, paste0(fig$id, ".png")), p,
                  width = fig$width, height = fig$height, dpi = 130,
                  bg = "white")
  specs[[fig$id]] <- serialize_value(p)
}

con <- gzfile(file.path(out, "specs.json.gz"), "wt")
writeLines(jsonlite::toJSON(specs, auto_unbox = TRUE, null = "null",
                            na = "string", digits = NA), con)
close(con)
message("\nwrote ", length(cat$figures), " figure(s) to ", out)

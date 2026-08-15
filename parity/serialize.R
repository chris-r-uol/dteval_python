#############################################################
# parity/serialize.R
#############################################################
#
# Lossless, diffable serialisation of R values for the Python parity harness.
#
# Design notes
# ------------
# * Doubles are written as `sprintf("%.17g", x)` STRINGS. 17 significant digits
#   round-trips every IEEE double exactly, so nothing is lost, and unlike a hex
#   float the output still reads like a number in a diff. NA / NaN / Inf / -Inf
#   are written as those literal tokens so they stay distinguishable -- note
#   that R's NA_real_ and NaN are different values here and must remain so.
# * Every column carries its R class, because column type is part of what we
#   are asserting parity on (integer vs numeric, factor levels and their order,
#   Date vs numeric).
# * jsonlite's default numeric handling is lossy; we never hand it a double.
#
# The Python side reads this with parity/compare.py.

DTE_SER_VERSION <- 1L

.ser_dbl <- function(x) {
  out <- sprintf("%.17g", x)
  out[is.nan(x)] <- "NaN"
  out[!is.na(x) & is.infinite(x) & x > 0] <- "Inf"
  out[!is.na(x) & is.infinite(x) & x < 0] <- "-Inf"
  # is.na() is TRUE for NaN too, so test for real NA last and explicitly.
  out[is.na(x) & !is.nan(x)] <- "NA"
  as.character(out)
}

# Character NA cannot be encoded as a sentinel string: any sentinel we picked
# could legitimately appear in the data. So character-ish columns carry an
# explicit list of NA positions (0-based, to match the Python side) alongside
# their values.
.ser_chr <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x
}

.ser_na_idx <- function(x) {
  as.integer(which(is.na(x)) - 1L)
}

.ser_int <- function(x) {
  out <- as.character(x)
  out[is.na(x)] <- "NA"
  out
}

.ser_lgl <- function(x) {
  out <- ifelse(x, "TRUE", "FALSE")
  out[is.na(x)] <- "NA"
  as.character(out)
}

#' Serialise a single atomic vector (or column) to a typed list.
serialize_vector <- function(x) {
  cls <- class(x)[1]

  if (inherits(x, "Date")) {
    # Serialise the UNDERLYING numeric (days since 1970-01-01), not the
    # formatted date. R's Date is a double, so a fractional day is
    # representable; formatting to "%Y-%m-%d" would flatten any such difference
    # away and let a genuine divergence pass as a match. deseasonTubeData also
    # feeds as.numeric(.date) straight into its LOESS model, so the numeric is
    # the value that actually matters.
    # `iso` is informational only, so the fixture still reads like dates.
    return(list(rtype = "Date",
                values = .ser_dbl(unclass(x)),
                iso = .ser_chr(format(x, "%Y-%m-%d")),
                na = .ser_na_idx(x)))
  }
  if (inherits(x, "POSIXct") || inherits(x, "POSIXt")) {
    return(list(
      rtype = "POSIXct",
      tzone = if (is.null(attr(x, "tzone"))) "" else attr(x, "tzone"),
      values = .ser_chr(format(x, "%Y-%m-%dT%H:%M:%OS6", tz = "UTC")),
      na = .ser_na_idx(x)
    ))
  }
  if (is.factor(x)) {
    return(list(
      rtype = if (is.ordered(x)) "ordered" else "factor",
      levels = as.character(levels(x)),
      values = .ser_chr(as.character(x)),
      na = .ser_na_idx(x)
    ))
  }
  if (is.integer(x)) {
    return(list(rtype = "integer", values = .ser_int(x)))
  }
  if (is.logical(x)) {
    return(list(rtype = "logical", values = .ser_lgl(x)))
  }
  if (is.double(x)) {
    return(list(rtype = "numeric", values = .ser_dbl(x)))
  }
  if (is.character(x)) {
    return(list(rtype = "character", values = .ser_chr(x), na = .ser_na_idx(x)))
  }
  list(rtype = paste0("unsupported:", cls), values = .ser_chr(as.character(x)),
       na = .ser_na_idx(x))
}

#' Serialise any R value the parity harness may encounter.
serialize_value <- function(x) {
  if (is.null(x)) {
    return(list(type = "NULL"))
  }

  if (inherits(x, "ggplot")) {
    return(serialize_ggplot(x))
  }

  if (is.data.frame(x)) {
    x <- as.data.frame(x)
    cols <- lapply(names(x), function(nm) {
      c(list(name = nm), serialize_vector(x[[nm]]))
    })
    return(list(
      type = "data.frame",
      nrow = nrow(x),
      ncol = ncol(x),
      row_names = as.character(attr(x, "row.names")),
      columns = cols
    ))
  }

  if (is.matrix(x)) {
    return(list(
      type = "matrix",
      dim = as.integer(dim(x)),
      dimnames = lapply(dimnames(x), function(d) if (is.null(d)) NULL else as.character(d)),
      data = serialize_vector(as.vector(x))
    ))
  }

  if (is.list(x)) {
    nms <- names(x)
    return(list(
      type = "list",
      names = if (is.null(nms)) NA_character_ else nms,
      values = unname(lapply(x, serialize_value))
    ))
  }

  if (is.atomic(x)) {
    v <- serialize_vector(x)
    nms <- names(x)
    return(c(
      list(type = "vector", length = length(x)),
      v,
      list(vnames = if (is.null(nms)) NA_character_ else as.character(nms))
    ))
  }

  list(type = paste0("unsupported:", class(x)[1]))
}

#' Serialise a ggplot by the decisions DTEval made, not by ggplot2's internals.
#'
#' What is compared, and why
#' -------------------------
#' For each layer: the geom, the *input* data frame DTEval handed to it, which
#' columns it mapped to which aesthetics, and which fixed parameters it set.
#' Plus the plot-level labels, facet specification and palette.
#'
#' Deliberately NOT ggplot_build() output. That would compare ggplot2's
#' internal columns (PANEL, group, flipped_aes, computed stat columns) against
#' whatever the Python plotting library happens to produce -- two libraries'
#' implementation details, not DTEval's behaviour. The input data is the better
#' contract precisely because everything DTEval computes for itself (the band
#' quantiles from calcTubeStat, the fitted values from fitTubeModel, the LOESS
#' bounds in testTubePrecision) is already in it, and is compared bit-exactly.
#'
#' `built_summary` keeps a light fingerprint of ggplot_build() -- row counts and
#' the panel count -- so a structural change in the rendered result still shows
#' up, without pinning us to ggplot2's column layout.
#' Content hash of a serialised value, for de-duplicating repeated layer data.
digest_layer <- function(x) {
  txt <- jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "string",
                          digits = NA)
  tf <- tempfile(); on.exit(unlink(tf))
  writeLines(as.character(txt), tf)
  unname(tools::md5sum(tf))
}

serialize_ggplot <- function(p) {
  lay <- function(i) {
    L <- p$layers[[i]]
    # A layer either carries its own data or inherits the plot's.
    dat <- L$data
    if (inherits(dat, "waiver") || is.null(dat)) dat <- p$data
    mapping <- tryCatch(
      vapply(L$mapping, rlang::as_label, character(1)),
      error = function(e) character(0)
    )
    params <- tryCatch(
      lapply(L$aes_params, function(v) paste(as.character(v), collapse = ",")),
      error = function(e) list()
    )
    stat_params <- tryCatch(
      lapply(L$stat_params, function(v) paste(as.character(v), collapse = ",")),
      error = function(e) list()
    )
    geom_params <- tryCatch(
      lapply(L$geom_params, function(v) paste(as.character(v), collapse = ",")),
      error = function(e) list()
    )
    list(
      index = i,
      geom = class(L$geom)[1],
      stat = class(L$stat)[1],
      position = class(L$position)[1],
      mapping = as.list(mapping),
      mapping_names = names(mapping),
      aes_params = params,
      stat_params = stat_params,
      geom_params = geom_params,
      data = serialize_value(as.data.frame(dat))
    )
  }
  layers <- if (length(p$layers)) lapply(seq_along(p$layers), lay) else list()

  # Layers very often share a data frame -- testTubePrecision hands the same
  # 7,500-row table to its points layer and the same sorted copy to all three
  # LOESS paths. Serialising each in full made one fixture 17 MB. Emit the
  # first occurrence and reference it thereafter; compare.py resolves refs, so
  # the comparison is unchanged.
  seen <- list()
  for (i in seq_along(layers)) {
    key <- digest_layer(layers[[i]]$data)
    hit <- match(key, unlist(seen))
    if (!is.na(hit)) {
      layers[[i]]$data <- list(type = "ref", layer = hit)
    }
    seen[[i]] <- key
  }

  facet <- class(p$facet)[1]
  facet_vars <- tryCatch(
    as.character(unlist(lapply(p$facet$params$facets, rlang::as_label))),
    error = function(e) character(0)
  )
  if (length(facet_vars) == 0) {
    facet_vars <- tryCatch(
      as.character(c(
        unlist(lapply(p$facet$params$rows, rlang::as_label)),
        unlist(lapply(p$facet$params$cols, rlang::as_label))
      )),
      error = function(e) character(0)
    )
  }

  built_summary <- tryCatch({
    b <- ggplot2::ggplot_build(p)
    list(n_layers = length(b$data),
         layer_rows = vapply(b$data, nrow, 1L),
         n_panels = length(unique(b$data[[1]]$PANEL)))
  }, error = function(e) list(error = conditionMessage(e)))

  scales <- tryCatch(
    vapply(p$scales$scales, function(s) class(s)[1], character(1)),
    error = function(e) character(0)
  )
  scale_aes <- tryCatch(
    unlist(lapply(p$scales$scales, function(s) paste(s$aesthetics, collapse = ","))),
    error = function(e) character(0)
  )
  palettes <- tryCatch(
    lapply(p$scales$scales, function(s) {
      v <- tryCatch(s$palette.cache, error = function(e) NULL)
      if (is.null(v)) v <- tryCatch(environment(s$palette)$values, error = function(e) NULL)
      if (is.null(v)) NA_character_ else as.character(v)
    }),
    error = function(e) list()
  )

  list(
    type = "ggplot",
    labels = lapply(p$labels, function(v) as.character(v)[1]),
    facet = facet,
    facet_vars = facet_vars,
    mapping = as.list(tryCatch(vapply(p$mapping, rlang::as_label, character(1)),
                               error = function(e) character(0))),
    mapping_names = names(p$mapping),
    scales = scales,
    scale_aes = scale_aes,
    palettes = palettes,
    built_summary = built_summary,
    layers = layers
  )
}

#' Write a serialised case to parity/fixtures/<id>.json
write_fixture <- function(id, value, dir = "parity/fixtures", meta = list()) {
  path <- file.path(dir, paste0(gsub("[/ ]", "__", id), ".json.gz"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  payload <- list(
    ser_version = DTE_SER_VERSION,
    id = id,
    meta = meta,
    value = serialize_value(value)
  )
  txt <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null",
                          na = "string", pretty = TRUE, digits = NA)

  # gzip, because these are large and highly repetitive. gzfile() writes a real
  # gzip container (memCompress(type="gzip") emits a bare zlib stream, which
  # standard tools cannot read) and does so deterministically -- no embedded
  # timestamp -- so identical content still yields identical bytes. That is what
  # keeps `git diff --exit-code` usable as the fixture-staleness gate in CI.
  con <- gzfile(path, "wt")
  writeLines(txt, con)
  close(con)
  invisible(path)
}

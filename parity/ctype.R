# Shared: pin LC_CTYPE to a UTF-8 locale.
#
# LC_COLLATE, LC_TIME and LC_NUMERIC were pinned from the start; LC_CTYPE was
# not, and it decides how R renders the non-ASCII site names in dt.brd. Under a
# C locale, load() translates the latin1 strings to UTF-8 and serialisation then
# escapes them as "<U+2018>"; under a UTF-8 locale the characters survive as
# themselves. Both are valid, but only one can be the fixture, and leaving it to
# the ambient environment meant the answer depended on whose shell ran it.
#
# UTF-8 is the pinned choice: the quotes in "Outside 'Vapes and Phones' shop"
# are real characters in the source data, and a frontend rendering "<U+2018>"
# to a user would be plainly wrong.
dte_pin_ctype <- function() {
  for (loc in c("C.UTF-8", "en_US.UTF-8", "UTF-8")) {
    if (nzchar(suppressWarnings(Sys.setlocale("LC_CTYPE", loc)))) {
      return(invisible(loc))
    }
  }
  stop("no UTF-8 locale available for LC_CTYPE; tried C.UTF-8, en_US.UTF-8, UTF-8",
       call. = FALSE)
}

"""Number formatting and rounding, R's way where it is visible in output.

Scope
-----
Only two things here affect what a user sees:

* :func:`signif` -- report strings embed ``signif(x, 4)`` verbatim, e.g.
  ``"mean: 25.69 (6.522 to 56.88)"``;
* :func:`as_character` -- ``.location`` is built as ``"{lat,lon}"``, so the
  number becomes part of a string.

An earlier revision reproduced R's ``format.c`` in full -- ``scientific()``,
``formatReal()``, ``EncodeReal()``, ``R_pow_di`` -- roughly 400 lines, so that
``.sample_id``'s pasted grouping key matched R byte for byte. That key is now
built from the ``(lat, lon, start, end)`` tuple directly, which groups
identically without any string round trip, so the transliteration is gone. See
docs/parity.md.

R prints a double with 15 significant digits where Python's ``repr`` uses the
shortest round-trip form; for values that have been through ``signif`` the two
agree, which covers the report strings.
"""

from __future__ import annotations

import math

__all__ = ["as_character", "signif"]


def signif(x: float, digits: int = 6) -> float:
    """R's ``signif()`` -- round to ``digits`` significant digits."""
    if x != x or math.isinf(x) or x == 0.0:
        return x
    if digits < 1:
        digits = 1
    magnitude = math.floor(math.log10(abs(x)))
    factor = 10.0 ** (digits - 1 - magnitude)
    return round(x * factor) / factor


def as_character(x: float | int | bool | None) -> str:
    """R's ``as.character()`` for a numeric scalar.

    Integers and booleans render as R does. Floats use Python's shortest
    round-trip representation rather than R's 15 significant digits -- the two
    agree for the rounded values that reach user-visible output, and nothing
    depends on the string for grouping any more.
    """
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, int):
        return str(x)
    value = float(x)
    if value != value:
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0.0:
        return "0"  # collapse -0.0, as R does
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)

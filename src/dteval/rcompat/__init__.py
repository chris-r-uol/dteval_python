"""R-semantics primitives.

These reproduce the specific behaviours of R that DTEval's *values* depend on --
number formatting, collation order, factor level construction, date handling,
quantile type 7, R's RNG stream -- so that the rest of the port can be written
as a direct translation of the R source without having to reason about
language differences at every line.

Each module is verified against R directly (see ``tests/test_rcompat_*.py``)
before anything is built on top of it.
"""

from __future__ import annotations

from dteval.rcompat.numfmt import as_character, signif

__all__ = ["as_character", "signif"]

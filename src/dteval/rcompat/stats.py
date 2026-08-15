"""R statistical primitives used by DTEval.

Thin wrappers over numpy/scipy, so R's conventions are written down once rather
than assumed at each of the dozens of call sites. What they capture is
*semantics*, not the last bit:

* ``sd``/``var`` use the n-1 denominator and give ``NA`` below two values;
* ``quantile`` is type 7 (R's default), which is numpy's ``linear``;
* ``median`` averages the two middle values;
* ``cut`` builds its interval labels with C's ``%.*g`` at ``dig.lab`` digits,
  which is why R's break labels read ``(100,1e+03]`` and not ``(100,1000]``.

Earlier revisions reproduced R's floating-point *arithmetic* as well -- its
two-pass mean, the FMA contraction in its variance loop, hand-ported ``qnorm``
and ``qt``. That bought agreement in the 16th digit on measurements carrying a
10% error bar, at the cost of ~520 lines of transliterated Fortran, Python-loop
accumulation and a dependence on ``long double`` being 64-bit. It was dropped;
see docs/parity.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as _sps

from dteval.rcompat.numfmt import signif

__all__ = [
    "is_finite",
    "qnorm",
    "qt",
    "r_cut",
    "r_max",
    "r_mean",
    "r_median",
    "r_min",
    "r_quantile",
    "r_sd",
    "r_var",
    "signif",
]


def _clean(x, na_rm: bool) -> np.ndarray:
    a = pd.Series(x).to_numpy(dtype="float64", na_value=np.nan)
    return a[~np.isnan(a)] if na_rm else a


def r_mean(x, na_rm: bool = False) -> float:
    """R's ``mean()``."""
    a = _clean(x, na_rm)
    return float(np.mean(a)) if a.size else np.nan


def r_median(x, na_rm: bool = False) -> float:
    """R's ``median()`` -- the mean of the two middle values when n is even."""
    a = _clean(x, na_rm)
    if a.size == 0 or np.isnan(a).any():
        return np.nan
    return float(np.median(a))


def r_var(x, na_rm: bool = False) -> float:
    """R's ``var()`` -- the n-1 denominator, ``NA`` below two values."""
    a = _clean(x, na_rm)
    if a.size < 2:
        return np.nan
    return float(np.var(a, ddof=1))


def r_sd(x, na_rm: bool = False) -> float:
    """R's ``sd()`` -- ``sqrt(var(x))``, ``NA`` below two values."""
    v = r_var(x, na_rm)
    return float(np.sqrt(v)) if v == v else np.nan


def r_min(x, na_rm: bool = False) -> float:
    a = _clean(x, na_rm)
    return float(np.min(a)) if a.size else math.inf


def r_max(x, na_rm: bool = False) -> float:
    a = _clean(x, na_rm)
    return float(np.max(a)) if a.size else -math.inf


def r_quantile(x, probs, na_rm: bool = False, type: int = 7) -> np.ndarray:
    """R's ``quantile()``, type 7 (R's default) -- numpy's ``linear`` method.

    The wrapper exists to make the type explicit and to reject the other eight
    loudly rather than silently computing a different quantile.
    """
    if type != 7:
        raise NotImplementedError(
            f"quantile type {type} is not implemented; DTEval only uses R's default (7)"
        )
    a = _clean(x, na_rm)
    probs = np.atleast_1d(np.asarray(probs, dtype="float64"))
    if a.size == 0 or np.isnan(a).any():
        return np.full(probs.shape, np.nan)
    return np.quantile(a, probs, method="linear")


def is_finite(x) -> np.ndarray:
    """R's ``is.finite()`` -- False for NA, NaN and the infinities."""
    a = pd.Series(x).to_numpy(dtype="float64", na_value=np.nan)
    return np.isfinite(a)




def qt(p, df, lower_tail: bool = True) -> np.ndarray:
    """R's ``qt()``. ``df <= 0`` gives NaN, as R does (with a warning)."""
    p = np.atleast_1d(np.asarray(p, dtype="float64"))
    df = np.atleast_1d(np.asarray(df, dtype="float64"))
    p, df = np.broadcast_arrays(p, df)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(df) & (df > 0)
    if np.any(ok):
        out[ok] = _sps.t.ppf(p[ok], df[ok]) if lower_tail else _sps.t.isf(p[ok], df[ok])
    return out


def qnorm(p, mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True) -> np.ndarray:
    """R's ``qnorm()``."""
    p = np.atleast_1d(np.asarray(p, dtype="float64"))
    if lower_tail:
        return _sps.norm.ppf(p, loc=mean, scale=sd)
    return _sps.norm.isf(p, loc=mean, scale=sd)


def _fmt_g(x: float, digits: int) -> str:
    """C's ``%.*g``, which is what R's ``formatC`` uses to build cut labels."""
    return "%.*g" % (digits, x)


def r_cut(
    x,
    breaks,
    labels=None,
    right: bool = True,
    include_lowest: bool = False,
    dig_lab: int = 3,
) -> pd.Series:
    """R's ``cut()`` for numeric breaks.

    Labels are ``(lo,hi]`` built with ``%.*g`` at ``dig.lab`` significant
    digits, so 1000 renders as ``1e+03`` -- matching R rather than looking like
    a bug. ``dig.lab`` is increased (as R does) until the labels are unique.
    """
    a = pd.Series(x).to_numpy(dtype="float64", na_value=np.nan)
    br = np.asarray(breaks, dtype="float64")
    if br.size < 2:
        raise ValueError("'breaks' must have at least two elements")
    br = np.sort(br)

    if labels is None:
        dl = dig_lab
        while dl < 12:
            ch = [_fmt_g(b, dl) for b in br]
            if len(set(ch)) == len(ch):
                break
            dl += 1
        ch = [_fmt_g(b, dl) for b in br]
        if right:
            labs = [f"({ch[i]},{ch[i + 1]}]" for i in range(len(ch) - 1)]
        else:
            labs = [f"[{ch[i]},{ch[i + 1]})" for i in range(len(ch) - 1)]
    elif labels is False:
        labs = [str(i + 1) for i in range(br.size - 1)]
    else:
        labs = [str(v) for v in labels]

    codes = np.full(a.shape, -1, dtype="int64")
    for i in range(br.size - 1):
        lo, hi = br[i], br[i + 1]
        if right:
            hit = (a > lo) & (a <= hi)
            if include_lowest and i == 0:
                hit |= a == lo
        else:
            hit = (a >= lo) & (a < hi)
            if include_lowest and i == br.size - 2:
                hit |= a == hi
        codes[hit] = i
    codes[np.isnan(a)] = -1
    return pd.Series(pd.Categorical.from_codes(codes, categories=labs, ordered=True))

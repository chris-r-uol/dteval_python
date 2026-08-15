"""R statistical primitives used by DTEval.

Most of these agree with numpy/scipy defaults; they are wrapped anyway so the
assumption is written down once and tested against R, rather than assumed at
each of the dozens of call sites.

The one that genuinely differs is :func:`r_cut`, whose interval labels are
built with C's ``%.*g`` at ``dig.lab`` digits -- which is why R's break labels
read ``(100,1e+03]`` rather than ``(100,1000]``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dteval.rcompat.numfmt import signif
from dteval.rcompat.rmath import qnorm, qt

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


def _seq_sum(a: np.ndarray) -> float:
    """Left-to-right summation, as C does it.

    numpy's ``sum`` uses pairwise summation, which is more accurate than a
    plain loop and therefore *disagrees with R in the last ulp*. ``cumsum``
    accumulates sequentially, matching R's C loop exactly.
    """
    if a.size == 0:
        return 0.0
    return float(np.cumsum(a)[-1])


def r_mean(x, na_rm: bool = False) -> float:
    """R's ``mean()`` -- port of ``rsum`` in ``src/main/summary.c``.

    R does not compute ``sum(x)/n``. It computes that, then makes a second pass
    to correct the accumulated rounding error:

        s = sum(x)/n;  if finite:  s += sum(x - s)/n

    Skipping the correction changes the last bit of nearly every group mean,
    which then propagates through every downstream statistic.
    """
    a = _clean(x, na_rm)
    n = a.size
    if n == 0:
        return np.nan
    s = _seq_sum(a) / n
    if math.isfinite(s):
        t = _seq_sum(a - s)
        s = s + t / n
    return float(s)


def r_median(x, na_rm: bool = False) -> float:
    """R's ``median()`` -- the mean of the two middle values when n is even."""
    a = _clean(x, na_rm)
    if a.size == 0 or np.isnan(a).any():
        return np.nan
    a = np.sort(a)
    n = a.size
    half = (n + 1) // 2
    if n % 2 == 1:
        return float(a[half - 1])
    return r_mean(a[half - 1 : half + 1])


def r_var(x, na_rm: bool = False) -> float:
    """R's ``var()`` -- the two-pass form from ``src/library/stats/src/cov.c``.

    It reuses the *refined* mean (:func:`r_mean`), then accumulates squared
    deviations and divides by ``n - 1``.

    The accumulation uses :func:`math.fma` deliberately. R's inner loop is
    ``sum += (LDOUBLE)(x[k] - xm) * (x[k] - xm)``, which the C compiler
    contracts into a single fused multiply-add under the default
    ``-ffp-contract=on``. An FMA rounds once where a separate multiply and add
    round twice, so writing it the obvious way disagrees with R in the last ulp
    on roughly a fifth of real replicate groups.

    Measured against R 4.6.1 (conda, aarch64) -- see
    ``tests/test_rcompat_stats.py``, which re-checks both figures:

    * DTEval's actual domain (2875 co-located replicate groups from ``dt.brd``,
      n = 2/3/5): FMA matches **2875/2875**, plain sequential only 2262.
    * An adversarial synthetic set (n up to 60, values spanning 13 orders of
      magnitude): FMA matches 312/400 and sequential 367/400, with 6 samples
      matching neither. At that size the compiled loop evidently takes a
      different shape, so 1-ulp disagreement is possible for wide-ranging
      inputs with large n. ``docs/parity.md`` records this.
    """
    a = _clean(x, na_rm)
    n = a.size
    if n < 2:
        return np.nan
    xm = r_mean(a)
    total = 0.0
    for v in a.tolist():
        d = v - xm
        total = math.fma(d, d, total)
    return float(total / (n - 1))


def r_sd(x, na_rm: bool = False) -> float:
    """R's ``sd()`` -- ``sqrt(var(x))``, ``NA`` for fewer than two values."""
    v = r_var(x, na_rm)
    return float(np.sqrt(v)) if v == v else np.nan


def r_min(x, na_rm: bool = False) -> float:
    a = _clean(x, na_rm)
    return float(np.min(a)) if a.size else math.inf


def r_max(x, na_rm: bool = False) -> float:
    a = _clean(x, na_rm)
    return float(np.max(a)) if a.size else -math.inf


def r_quantile(x, probs, na_rm: bool = False, type: int = 7) -> np.ndarray:
    """R's ``quantile()``, type 7 (R's default).

    numpy's ``linear`` method is the same quantile mathematically but evaluates
    it as ``lo + (hi - lo) * h``; R evaluates ``(1 - h) * lo + h * hi``. Those
    round differently, so the formula is written out R's way here.
    """
    if type != 7:
        raise NotImplementedError(
            f"quantile type {type} is not implemented; DTEval only uses R's default (7)"
        )
    a = _clean(x, na_rm)
    probs = np.atleast_1d(np.asarray(probs, dtype="float64"))
    if a.size == 0:
        return np.full(probs.shape, np.nan)
    if np.isnan(a).any():
        return np.full(probs.shape, np.nan)

    a = np.sort(a)
    n = a.size
    index = 1.0 + (n - 1) * probs
    lo = np.floor(index)
    hi = np.ceil(index)
    out = a[(lo - 1).astype("int64")]
    hi_val = a[(hi - 1).astype("int64")]
    h = index - lo
    move = (index > lo) & (hi_val != out)
    out = out.astype("float64").copy()
    out[move] = (1 - h[move]) * out[move] + h[move] * hi_val[move]
    return out


def is_finite(x) -> np.ndarray:
    """R's ``is.finite()`` -- False for NA, NaN and the infinities."""
    a = pd.Series(x).to_numpy(dtype="float64", na_value=np.nan)
    return np.isfinite(a)




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

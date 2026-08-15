"""Ports of R's ``nmath`` quantile functions.

``testTubePrecision`` builds its confidence bounds from ``qt`` (method 2) and
``qnorm`` (method 4), and those bounds are then LOESS-smoothed and printed in
the report, so a last-ulp difference propagates into the visible output. R and
scipy use different algorithms and disagree in the 16th digit, which is enough
to fail a bit-exact comparison -- hence these ports.

* :func:`qnorm` is Wichura's AS 241, exactly as ``src/nmath/qnorm.c``.
* :func:`qt` follows ``src/nmath/qt.c``: the closed forms for ``df ~= 1``,
  ``df ~= 2`` and ``df > 1e20`` are exact ports, and the general branch is
  Hill's (1981) expansion followed by R's 2-term Taylor refinement. The
  refinement needs ``pt``/``dt``; porting R's incomplete beta as well would be a
  disproportionate amount of code, so scipy supplies those two densities. The
  loop iterates to ``|x| <= 1e-14 * |q|``, so it converges to the same fixed
  point -- see ``tests/test_rcompat_rmath.py``, which measures the agreement
  across the df range rather than assuming it.

DTEval's own use is ``qt(0.025, df = n - 1, lower.tail = FALSE)`` where ``n`` is
the replicate count, so the default path (``n = 3``) lands on the exact ``df ==
2`` closed form.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as _sps

__all__ = ["qnorm", "qnorm_scalar", "qt", "qt_scalar"]

_DBL_EPSILON = 2.220446049250313e-16
_DBL_MIN = 2.2250738585072014e-308
_DBL_MAX = 1.7976931348623157e308
_M_SQRT2 = 1.4142135623730951
_M_PI_2 = 1.5707963267948966
_M_LN2 = 0.6931471805599453


def _horner(r: float, coef: tuple[float, ...]) -> float:
    """Horner evaluation in C's association order: ``((c0*r + c1)*r + c2)...``.

    Each step uses :func:`math.fma` because the C compiler contracts
    ``acc * r + c`` into a fused multiply-add under the default
    ``-ffp-contract=on``. Evaluating it as a separate multiply and add rounds
    twice and costs an ulp in about a third of the qnorm domain.
    """
    acc = coef[0]
    for c in coef[1:]:
        acc = math.fma(acc, r, c)
    return acc


# AS 241 coefficients, highest order first (src/nmath/qnorm.c).
# Central region, |p - 0.5| <= 0.425:
_A_NUM = (
    2509.0809287301226727, 33430.575583588128105, 67265.770927008700853,
    45921.953931549871457, 13731.693765509461125, 1971.5909503065514427,
    133.14166789178437745, 3.387132872796366608,
)
_A_DEN = (
    5226.495278852854561, 28729.085735721942674, 39307.89580009271061,
    21213.794301586595867, 5394.1960214247511077, 687.1870074920579083,
    42.313330701600911252, 1.0,
)
# Intermediate tail, r = sqrt(-log(min(p, 1-p))) <= 5:
_B_NUM = (
    7.7454501427834140764e-4, 0.0227238449892691845833, 0.24178072517745061177,
    1.27045825245236838258, 3.64784832476320460504, 5.7694972214606914055,
    4.6303378461565452959, 1.42343711074968357734,
)
_B_DEN = (
    1.05075007164441684324e-9, 5.475938084995344946e-4, 0.0151986665636164571966,
    0.14810397642748007459, 0.68976733498510000455, 1.6763848301838038494,
    2.05319162663775882187, 1.0,
)
# Extreme tail, 5 < r <= 27:
_C_NUM = (
    2.01033439929228813265e-7, 2.71155556874348757815e-5, 0.0012426609473880784386,
    0.026532189526576123093, 0.29656057182850489123, 1.7848265399172913358,
    5.4637849111641143699, 6.6579046435011037772,
)
_C_DEN = (
    2.04426310338993978564e-15, 1.4215117583164458887e-7, 1.8463183175100546818e-5,
    7.868691311456132591e-4, 0.0148753612908506148525, 0.13692988092273580531,
    0.59983220655588793769, 1.0,
)


def _d_lval(p: float, lower_tail: bool) -> float:
    """R's ``R_D_Lval``: ``p`` on the lower tail, ``0.5 - p + 0.5`` otherwise."""
    return p if lower_tail else 0.5 - p + 0.5


def _d_cval(p: float, lower_tail: bool) -> float:
    """R's ``R_D_Cval``: the complement, written to keep precision near 1."""
    return 0.5 - p + 0.5 if lower_tail else p


def qnorm_scalar(p: float, mu: float = 0.0, sigma: float = 1.0, lower_tail: bool = True) -> float:
    """R's ``qnorm()`` -- Wichura AS 241 (``src/nmath/qnorm.c``)."""
    if p != p or mu != mu or sigma != sigma:
        return p + mu + sigma
    if p < 0 or p > 1:
        return math.nan
    if p == 0:
        return -math.inf if lower_tail else math.inf
    if p == 1:
        return math.inf if lower_tail else -math.inf
    if sigma < 0:
        return math.nan
    if sigma == 0:
        return mu

    # R's R_DT_qIv. The complement is written 0.5 - p + 0.5, not 1 - p: that
    # form loses less precision for p near 1, and using the obvious expression
    # instead costs an ulp in the result.
    p_ = p if lower_tail else _d_cval(p, True)
    q = p_ - 0.5

    if abs(q) <= 0.425:
        # C: r = .180625 - q * q, which the compiler contracts to a single FMA.
        r = math.fma(-q, q, 0.180625)
        val = q * _horner(r, _A_NUM) / _horner(r, _A_DEN)
        return mu + sigma * val

    # R_DT_CIv(p) = R_D_Cval(p), which takes the ORIGINAL p and the caller's
    # lower_tail -- not the already-complemented p_. Feeding it p_ recomputes
    # 0.5 - (0.5 - p + 0.5) + 0.5 and loses a bit on the round trip.
    lp = math.log(_d_cval(p, lower_tail) if q > 0 else p_)
    r = math.sqrt(-lp)

    if r <= 5.0:
        r += -1.6
        val = _horner(r, _B_NUM) / _horner(r, _B_DEN)
    else:
        r += -5.0
        val = _horner(r, _C_NUM) / _horner(r, _C_DEN)

    if q < 0.0:
        val = -val
    return mu + sigma * val


def qt_scalar(p: float, ndf: float, lower_tail: bool = True) -> float:
    """R's ``qt()`` -- port of ``src/nmath/qt.c`` (``log_p = FALSE`` only)."""
    eps = 1e-12

    if p != p or ndf != ndf:
        return p + ndf
    if p < 0 or p > 1:
        return math.nan
    if p == 0:
        return -math.inf if lower_tail else math.inf
    if p == 1:
        return math.inf if lower_tail else -math.inf
    if ndf <= 0:
        return math.nan

    if ndf < 1:
        # R inverts pt() by bisection here.
        accu, Eps = 1e-13, 1e-11
        pp_target = p if lower_tail else 1.0 - p
        if pp_target > 1 - _DBL_EPSILON:
            return math.inf
        pp = min(1 - _DBL_EPSILON, pp_target * (1 + Eps))
        ux = 1.0
        while ux < _DBL_MAX and _sps.t.cdf(ux, ndf) < pp:
            ux *= 2
        pp = pp_target * (1 - Eps)
        lx = -1.0
        while lx > -_DBL_MAX and _sps.t.cdf(lx, ndf) > pp:
            lx *= 2
        nx = 0.5 * (lx + ux)
        it = 0
        while (ux - lx) / abs(nx) > accu and it < 1000:
            nx = 0.5 * (lx + ux)
            if _sps.t.cdf(nx, ndf) > pp_target:
                ux = nx
            else:
                lx = nx
            it += 1
        return 0.5 * (lx + ux)

    if ndf > 1e20:
        return qnorm_scalar(p, 0.0, 1.0, lower_tail)

    P = p  # R_D_qIv(p) with log_p = FALSE
    neg = ((not lower_tail) or P < 0.5) and (lower_tail or P > 0.5)

    # P becomes 2 * min(P', 1 - P'), built through R_D_Lval / R_D_Cval so the
    # complement keeps full precision near 1.
    P = 2 * (_d_lval(p, lower_tail) if neg else _d_cval(p, lower_tail))

    if abs(ndf - 2) < eps:  # df ~= 2
        if P > _DBL_MIN:
            if 3 * P < _DBL_EPSILON:
                q = 1 / math.sqrt(P)
            elif P > 0.9:
                q = (1 - P) * math.sqrt(2 / (P * (2 - P)))
            else:
                q = math.sqrt(2 / (P * (2 - P)) - 2)
        else:
            q = math.inf
    elif ndf < 1 + eps:  # df ~= 1: Cauchy
        if P == 1.0:
            q = 0.0
        elif P > 0:
            q = 1 / math.tan(math.pi * (P / 2.0))
        else:
            q = math.inf
    else:
        a = 1 / (ndf - 0.5)
        b = 48 / (a * a)
        c = ((20700 * a / b - 98) * a - 16) * a + 96.36
        d = ((94.5 / (b + c) - 3) / b + 1) * math.sqrt(a * _M_PI_2) * ndf
        x = 0.0
        y = math.pow(d * P, 2.0 / ndf)
        P_ok = y >= _DBL_EPSILON
        if not P_ok:
            log_P2 = math.log(P / 2.0)
            x = (math.log(d) + _M_LN2 + log_P2) / ndf
            y = math.exp(2 * x)

        if (ndf < 2.1 and P > 0.5) or y > 0.05 + a:
            # Asymptotic inverse expansion about normal
            x = qnorm_scalar(0.5 * P, 0.0, 1.0, True)
            y = x * x
            if ndf < 5:
                c += 0.3 * (ndf - 4.5) * (x + 0.6)
            c = (((0.05 * d * x - 5) * x - 7) * x - 2) * x + b + c
            y = (((((0.4 * y + 6.3) * y + 36) * y + 94.5) / c - y - 3) / b + 1) * x
            y = math.expm1(a * y * y)
            q = math.sqrt(ndf * y)
        elif not P_ok and x < -_M_LN2 * 53:
            q = math.sqrt(ndf) * math.exp(-x)
        else:
            y = (
                (1 / (((ndf + 6) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2) * 3)
                 + 0.5 / (ndf + 4)) * y - 1
            ) * (ndf + 1) / (ndf + 2) + 1 / y
            q = math.sqrt(ndf * y)

        # Hill's 2-term Taylor refinement.
        M = abs(math.sqrt(_DBL_MAX / 2.0) - ndf)
        it = 0
        while it < 10:
            it += 1
            yy = _sps.t.pdf(q, ndf)
            if not yy > 0:
                break
            x = (_sps.t.sf(q, ndf) - P / 2) / yy
            if not math.isfinite(x) or abs(x) <= 1e-14 * abs(q):
                break
            F = q * (ndf + 1) / (2 * (q * q + ndf)) if abs(q) < M else (ndf + 1) / (
                2 * (q + ndf / q)
            )
            del_q = x * (1.0 + x * F)
            if math.isfinite(del_q) and math.isfinite(q + del_q):
                q += del_q
            elif math.isfinite(x) and math.isfinite(q + x):
                q += x
            else:
                break

    return -q if neg else q


def qnorm(p, mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True) -> np.ndarray:
    """Vectorised :func:`qnorm_scalar`."""
    arr = np.atleast_1d(np.asarray(p, dtype="float64"))
    return np.array([qnorm_scalar(float(v), mean, sd, lower_tail) for v in arr])


def qt(p, df, lower_tail: bool = True) -> np.ndarray:
    """Vectorised :func:`qt_scalar`; ``p`` and ``df`` broadcast against each other."""
    pa = np.atleast_1d(np.asarray(p, dtype="float64"))
    da = np.atleast_1d(np.asarray(df, dtype="float64"))
    pa, da = np.broadcast_arrays(pa, da)
    return np.array(
        [
            qt_scalar(float(pv), float(dv), lower_tail)
            for pv, dv in zip(pa.ravel(), da.ravel(), strict=True)
        ]
    ).reshape(pa.shape)

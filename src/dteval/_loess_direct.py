"""Multivariate LOESS with ``surface = "direct"``, ported from R's algorithm.

Why this is hand-written rather than delegated
----------------------------------------------
``scikit-misc`` 0.5.2 wraps the same netlib ``dloess`` R uses, and for a single
predictor it agrees with R to about 10 ulps. Its **multivariate** path does not
fit correctly at all -- with ``y = u`` exactly and an irrelevant second
predictor it returns a near-constant -- so it cannot be used for the two- and
three-predictor models ``fitTubeModel_loess`` builds.

``surface = "direct"`` is the tractable half of LOESS: the local fit is
evaluated at each point directly, with no kd-tree and no interpolation, so it
is plain tricube-weighted local polynomial regression. Measured against R over
a smooth two-predictor surface, fitted values agree to **2e-15 relative**
(a few ulps).

What is *not* reproduced exactly is ``se.fit``. R scales it by a residual
standard error derived from an approximate ``delta1`` (``statistics =
"1.approx"``), computed by the Fortran ``lowesa``/``ehg141``, which in turn
evaluates a hard-coded spline through ``ehg128`` -- 339 lines of tensor-product
blending. Using the exact hat matrix instead gives ``se.fit`` about 6e-4
relative from R's. The *fitted values* are unaffected; see docs/parity.md.

``surface = "interpolate"`` (R's default, and what ``deseasonTubeData`` uses
with two predictors) needs that same ``ehg128`` kd-tree machinery and is not
implemented here.
"""

from __future__ import annotations

import numpy as np

from dteval.rcompat.stats import r_var

__all__ = ["loess_direct", "normalise_predictors"]


def normalise_predictors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's ``loess(normalize = TRUE)`` scaling for two or more predictors.

    Each column is divided by the standard deviation of its 10%-trimmed sorted
    values, so predictors on wildly different scales (a day-of-year against a
    date, say) contribute comparably to the neighbourhood distance.
    """
    n, p = x.shape
    trim = int(np.ceil(0.1 * n))
    divisor = np.array(
        [np.sqrt(r_var(np.sort(x[:, j])[trim : n - trim])) for j in range(p)]
    )
    divisor[divisor == 0] = 1.0
    return x / divisor, divisor


def _design(deltas: np.ndarray, degree: int) -> np.ndarray:
    """Local polynomial terms about the evaluation point.

    Degree 2 with p predictors gives the full quadratic: an intercept, the p
    linear terms, then every square and cross-product -- matching R's local
    model when ``drop.square`` and ``parametric`` are left at their defaults.
    """
    p = deltas.shape[1]
    cols = [np.ones(len(deltas))]
    if degree >= 1:
        cols.extend(deltas[:, j] for j in range(p))
    if degree >= 2:
        cols.extend(
            deltas[:, j] * deltas[:, k] for j in range(p) for k in range(j, p)
        )
    return np.column_stack(cols)


def _kernel_row(xn: np.ndarray, x0: np.ndarray, q: int, span: float, degree: int):
    """The equivalent-kernel weights ``l(x0)``, so ``fit(x0) = l(x0) . y``."""
    dist = np.sqrt(((xn - x0) ** 2).sum(axis=1))
    bandwidth = np.sort(dist)[q - 1]
    if span > 1:
        # R widens the bandwidth for span > 1 rather than taking more neighbours.
        bandwidth = bandwidth * span ** (1.0 / xn.shape[1])

    if bandwidth <= 0:
        weights = (dist == 0).astype("float64")
    else:
        scaled = dist / bandwidth
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)

    design = _design(xn - x0, degree)
    weighted = design * weights[:, None]
    # pinv rather than solve: the local system is singular wherever a
    # neighbourhood is degenerate (all points collinear, or fewer neighbours
    # than terms), which R tolerates rather than failing on.
    return (weighted @ np.linalg.pinv(design.T @ weighted))[:, 0]


def loess_direct(
    x: np.ndarray,
    y: np.ndarray,
    newx: np.ndarray | None = None,
    span: float = 0.75,
    degree: int = 2,
    normalize: bool = True,
    se: bool = False,
):
    """Fit LOESS with ``surface="direct"`` and predict.

    Returns the fitted values, or ``(fitted, se_fit)`` when ``se`` is set.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    xn, divisor = (
        normalise_predictors(x) if (normalize and x.shape[1] > 1) else (x, np.ones(x.shape[1]))
    )
    n = len(xn)
    q = max(1, min(n, int(np.floor(n * span))))

    targets = xn if newx is None else np.asarray(newx, dtype="float64").reshape(-1, x.shape[1]) / divisor

    rows = np.empty((len(targets), n))
    for i, x0 in enumerate(targets):
        rows[i] = _kernel_row(xn, x0, q, span, degree)
    fitted = rows @ y

    if not se:
        return fitted

    # Residual scale. R divides by an approximate delta1; we use the exact
    # trace((I-L)'(I-L)), which differs by ~6e-4 relative -- see the module
    # docstring and docs/parity.md.
    hat = rows if newx is None else np.array(
        [_kernel_row(xn, x0, q, span, degree) for x0 in xn]
    )
    residual = y - hat @ y
    resid_op = np.eye(n) - hat
    delta1 = np.trace(resid_op.T @ resid_op)
    sigma = np.sqrt((residual**2).sum() / delta1)
    return fitted, sigma * np.sqrt((rows**2).sum(axis=1))

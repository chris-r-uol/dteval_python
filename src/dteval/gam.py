"""A tensor-product GAM -- the Python stand-in for ``mgcv::gam(y ~ te(...))``.

``fitTubeModel_gam`` fits ``[tube] ~ te(x1, x2, ...)`` with ``mgcv``'s defaults:
cubic-regression-spline marginals with ``k = 5`` knots, a tensor-product basis,
one wiggliness penalty per marginal direction, and smoothing parameters chosen
by GCV.

That is what is implemented here, directly on numpy and scipy. It is the one
place in the port where the answer is close rather than equal: mgcv's own
optimiser (a nested Newton scheme with its own reparameterisations and step
control) can settle on slightly different smoothing parameters than a general
optimiser given the same GCV score. The fitted surface tracks R's closely; the
measured deviation is recorded in docs/parity.md and pinned by tests.

Reference: Wood, *Generalized Additive Models: An Introduction with R* (2nd ed),
sections 5.3.1 (the ``cr`` basis) and 5.6 (tensor products).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = ["TensorGAM", "fit_tensor_gam"]

#: mgcv's default basis dimension per marginal for a 2-term ``te()``.
DEFAULT_K = 5


@dataclass
class TensorGAM:
    """A fitted tensor-product GAM, with enough state to predict with errors."""

    knots: list[np.ndarray]
    constraint: np.ndarray
    coef: np.ndarray
    cov: np.ndarray
    lambdas: np.ndarray
    scale: float
    edf: float

    def predict(self, x: np.ndarray, se: bool = False):
        """Predictions at ``x`` (n x d), optionally with standard errors."""
        design = _design(np.asarray(x, float), self.knots, self.constraint)
        fit = design @ self.coef
        if not se:
            return fit
        stderr = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, self.cov, design), 0))
        return fit, stderr


def fit_tensor_gam(x: np.ndarray, y: np.ndarray, k: int = DEFAULT_K) -> TensorGAM:
    """Fit ``y ~ te(x[:, 0], x[:, 1], ...)`` by penalised least squares with GCV."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[keep], y[keep]
    n, d = x.shape

    knots = [_knots(x[:, j], k) for j in range(d)]
    marginals = [_cr_basis(x[:, j], knots[j]) for j in range(d)]
    penalties = [_cr_penalty(knots[j]) for j in range(d)]

    full = _tensor(marginals)
    # One sum-to-zero constraint on the tensor, absorbed into the basis so the
    # intercept stays identifiable. mgcv does the same via a QR of the column
    # sums; the resulting fitted values are invariant to which basis is used.
    constraint = _null_space(full.mean(axis=0))
    design = np.column_stack([np.ones(n), full @ constraint])

    smooths = []
    for j in range(d):
        blocks = [penalties[i] if i == j else np.eye(len(knots[i])) for i in range(d)]
        block = blocks[0]
        for extra in blocks[1:]:
            block = np.kron(block, extra)
        block = constraint.T @ block @ constraint
        smooths.append(_pad(block))

    xtx = design.T @ design
    xty = design.T @ y
    yty = float(y @ y)

    def gcv(log_lambda):
        score, *_ = _penalised_fit(xtx, xty, yty, n, smooths, np.exp(log_lambda))
        return score

    start = np.zeros(d)
    best = optimize.minimize(gcv, start, method="Nelder-Mead",
                             options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    lambdas = np.exp(best.x)
    _, coef, cov, scale, edf = _penalised_fit(xtx, xty, yty, n, smooths, lambdas)

    return TensorGAM(knots=knots, constraint=constraint, coef=coef, cov=cov,
                     lambdas=lambdas, scale=scale, edf=edf)


def _penalised_fit(xtx, xty, yty, n, smooths, lambdas):
    """Solve the penalised normal equations and score them by GCV."""
    penalty = sum(lam * S for lam, S in zip(lambdas, smooths, strict=True))
    lhs = xtx + penalty
    # A ridge keeps the solve stable when a smoothing parameter runs away.
    lhs = lhs + np.eye(len(lhs)) * 1e-10 * np.trace(xtx) / len(lhs)
    inverse = np.linalg.inv(lhs)
    coef = inverse @ xty

    rss = yty - 2 * float(coef @ xty) + float(coef @ xtx @ coef)
    edf = float(np.trace(inverse @ xtx))
    residual_df = n - edf
    if residual_df <= 0:
        return np.inf, coef, inverse, np.inf, edf

    score = n * rss / residual_df**2
    scale = rss / residual_df
    return score, coef, inverse * scale, scale, edf


def _knots(v: np.ndarray, k: int) -> np.ndarray:
    """mgcv places ``cr`` knots at evenly spaced quantiles of the *unique* values.

    The distinction matters: tube coordinates repeat heavily (one location, many
    sampling periods), so quantiles of the raw column would bunch the knots
    around the busiest sites rather than spreading them over the domain.
    """
    return np.quantile(np.unique(v), np.linspace(0, 1, k))


def _cr_matrices(knots: np.ndarray):
    """``B`` and ``D`` from Wood section 5.3.1, mapping knot values to curvature."""
    k = len(knots)
    h = np.diff(knots)
    B = np.zeros((k - 2, k - 2))
    D = np.zeros((k - 2, k))
    for i in range(k - 2):
        D[i, i] = 1 / h[i]
        D[i, i + 1] = -1 / h[i] - 1 / h[i + 1]
        D[i, i + 2] = 1 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i > 0:
            B[i, i - 1] = h[i] / 6
        if i < k - 3:
            B[i, i + 1] = h[i + 1] / 6
    return B, D


def _cr_basis(v: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """The cubic-regression-spline basis: one column per knot value.

    Parameters are the spline's values *at the knots*, which is what makes the
    basis stable and the penalty cheap. Outside the knot range the spline is
    extended linearly, as mgcv does.
    """
    k = len(knots)
    B, D = _cr_matrices(knots)
    F = np.zeros((k, k))
    F[1:-1] = np.linalg.solve(B, D)

    h = np.diff(knots)
    j = np.clip(np.searchsorted(knots, v, side="right") - 1, 0, k - 2)
    hj = h[j]
    left = knots[j + 1] - v
    right = v - knots[j]

    out = np.zeros((len(v), k))
    rows = np.arange(len(v))
    out[rows, j] += left / hj
    out[rows, j + 1] += right / hj
    c_left = (left**3 / hj - hj * left) / 6
    c_right = (right**3 / hj - hj * right) / 6
    out += c_left[:, None] * F[j] + c_right[:, None] * F[j + 1]
    return out


def _cr_penalty(knots: np.ndarray) -> np.ndarray:
    """The integrated squared second derivative, ``D' B^-1 D``."""
    B, D = _cr_matrices(knots)
    return D.T @ np.linalg.solve(B, D)


def _tensor(marginals: list[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker product of the marginal bases."""
    out = marginals[0]
    for extra in marginals[1:]:
        out = (out[:, :, None] * extra[:, None, :]).reshape(len(out), -1)
    return out


def _null_space(v: np.ndarray) -> np.ndarray:
    """An orthonormal basis for the space orthogonal to ``v``."""
    q, _ = np.linalg.qr(v.reshape(-1, 1), mode="complete")
    return q[:, 1:]


def _pad(block: np.ndarray) -> np.ndarray:
    """Grow a penalty by one leading zero row/column, for the intercept."""
    out = np.zeros((len(block) + 1, len(block) + 1))
    out[1:, 1:] = block
    return out


def _design(x: np.ndarray, knots: list[np.ndarray], constraint: np.ndarray) -> np.ndarray:
    marginals = [_cr_basis(x[:, j], knots[j]) for j in range(x.shape[1])]
    return np.column_stack([np.ones(len(x)), _tensor(marginals) @ constraint])

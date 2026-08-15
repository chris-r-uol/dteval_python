"""LOESS, as DTEval uses it.

DTEval fits LOESS in three places: ``testTubePrecision`` (one predictor),
``deseasonTubeData`` (two -- day-of-year and date) and ``fitTubeModel_loess``
(two or more).

One implementation, one surface
-------------------------------
R offers two surfaces. ``surface = "direct"`` evaluates the local regression at
every point; ``surface = "interpolate"`` (R's default) builds a kd-tree and
blends between vertices, which is an approximation of ``direct`` made for
speed. This module always computes the **exact** local regression --
:mod:`dteval._loess_direct` -- and treats ``surface`` as advisory.

That choice is measured, not assumed:

* Against R's own ``surface="direct"``: agreement to ~2e-13.
* Against R's *default* ``interpolate``, the gap depends on how much data the
  kd-tree has to work with. On ``testTubePrecision``'s 7,527 points it is
  5e-12 relative -- invisible. On ``deseasonTubeData``'s smaller per-location
  groups it reaches 2.9 ug/m3, which is why that one case is non-gating.

Doing it this way also removes ``scikit-misc`` -- a compiled dependency whose
multivariate path does not fit correctly (see
:class:`MultivariateLoessUnavailable` in the git history for the reproducer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dteval._loess_direct import loess_direct

__all__ = ["LoessFit", "r_loess"]


@dataclass
class LoessFit:
    """A fitted LOESS model, mirroring what R's ``loess`` object offers."""

    x: np.ndarray
    y: np.ndarray
    span: float = 0.75
    degree: int = 2
    family: str = "gaussian"
    surface: str = "direct"
    normalize: bool = True
    #: Positions of the observations actually used, so callers can scatter
    #: results back into the original frame -- the idiom DTEval relies on after
    #: every fit (``d2$..fit[as.numeric(names(temp$fit))] <- temp$fit``).
    used_index: np.ndarray = field(default_factory=lambda: np.array([], dtype="int64"))

    @property
    def fitted(self) -> np.ndarray:
        return self.predict()

    def predict(self, newdata=None, se: bool = False):
        """R's ``predict.loess``.

        ``newdata=None`` returns the fitted values for the observations used in
        the fit; ``se=True`` returns ``(fit, se_fit)``.
        """
        target = None
        if newdata is not None:
            target = _as_predictor_matrix(newdata)
            if target.shape[1] != self.x.shape[1]:
                raise ValueError(
                    f"newdata has {target.shape[1]} predictor(s), "
                    f"model has {self.x.shape[1]}"
                )
        return loess_direct(
            self.x,
            self.y,
            target,
            span=self.span,
            degree=self.degree,
            normalize=self.normalize,
            se=se,
        )

    def scatter(self, values: np.ndarray, n: int) -> np.ndarray:
        """Place ``values`` back at their original row positions, NA elsewhere.

        Rows dropped for missing values stay NA, as R's ``na.omit`` leaves them.
        """
        out = np.full(n, np.nan)
        out[self.used_index] = values[: len(self.used_index)]
        return out


def _as_predictor_matrix(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype="float64")
    arr = np.asarray(x, dtype="float64")
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def r_loess(
    x,
    y,
    span: float = 0.75,
    degree: int = 2,
    family: str = "gaussian",
    surface: str = "direct",
    normalize: bool = True,
    cell: float = 0.2,
) -> LoessFit:
    """Fit LOESS with R's defaults for span, degree and family.

    ``surface`` and ``cell`` are accepted for call-site compatibility with R
    but do not change the fit -- the exact local regression is always computed.
    See the module docstring for the measured consequences.

    Rows with a missing predictor or response are dropped before fitting, as
    R's ``na.action = na.omit`` does; :meth:`LoessFit.scatter` puts results back
    in the right places.
    """
    xs = _as_predictor_matrix(x)
    ys = pd.Series(y).to_numpy(dtype="float64")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError(f"x has {xs.shape[0]} rows, y has {ys.shape[0]}")

    keep = ~(np.isnan(ys) | np.any(np.isnan(xs), axis=1))
    return LoessFit(
        x=xs[keep],
        y=ys[keep],
        span=span,
        degree=degree,
        family=family,
        surface=surface,
        normalize=normalize,
        used_index=np.flatnonzero(keep),
    )

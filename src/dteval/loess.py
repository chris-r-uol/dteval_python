"""R-compatible ``loess``.

DTEval fits LOESS in three places -- ``testTubePrecision`` (one predictor),
``deseasonTubeData`` (two: day-of-year and date) and ``fitTubeModel_loess``
(two or more, ``surface = "direct"``) -- so the fits are values, not
diagnostics, and have to reproduce R's.

Implementation
--------------
R's ``stats::loess`` is a thin wrapper over netlib's ``dloess`` (Cleveland's
C/Fortran). ``scikit-misc`` vendors the same code, so for a **single
predictor** it agrees with R to within about 10 ulps -- see
:data:`UNIVARIATE_AGREEMENT`.

Multivariate is a different story: ``scikit-misc`` 0.5.2 does not fit it
correctly, so this module refuses it rather than returning plausible-looking
wrong numbers. See :class:`MultivariateLoessUnavailable` for the reproducer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "UNIVARIATE_AGREEMENT",
    "LoessFit",
    "MultivariateLoessUnavailable",
    "r_loess",
]

#: Measured agreement with R 4.6.1 for a single predictor, in ulps, across
#: span/degree/surface combinations. Re-checked by tests/test_loess.py.
UNIVARIATE_AGREEMENT = 16


class MultivariateLoessUnavailable(NotImplementedError):
    """Raised for LOESS with more than one predictor.

    ``scikit-misc`` 0.5.2's multivariate path does not reproduce R, and does
    not appear to fit correctly at all. Reproducer: with ``y = u`` exactly
    (noiseless) and a second, irrelevant predictor ``v``,

    * one predictor  -> recovers ``y`` to 1e-13;
    * two predictors -> returns a near-constant (fitted sd 5.6 against ``y``'s
      30.1, worst error 60 over a range of 100), for both
      ``surface="interpolate"`` and ``surface="direct"``.

    Against R on a smooth two-predictor surface the relative error reaches 2.6.
    That is a wrong model, not a rounding difference, so the failure is raised
    rather than tolerated -- silently returning those numbers would corrupt
    ``deseasonTubeData`` and ``fitTubeModel_loess`` while looking plausible.
    """


class _DirectModel:
    """Adapter giving the native direct-surface fit the same predict() shape."""

    __slots__ = ("x", "y", "span", "degree", "normalize")

    def __init__(self, x, y, span, degree, normalize):
        self.x, self.y = x, y
        self.span, self.degree, self.normalize = span, degree, normalize

    def predict(self, newdata, stderror: bool = False):
        from dteval._loess_direct import loess_direct

        nd = None if newdata is None else np.asarray(newdata, dtype="float64")
        out = loess_direct(
            self.x, self.y, nd, span=self.span, degree=self.degree,
            normalize=self.normalize, se=stderror,
        )
        if stderror:
            values, stderr = out
            return _Prediction(values, stderr)
        return _Prediction(out, None)


class _Prediction:
    __slots__ = ("values", "stderr")

    def __init__(self, values, stderr):
        self.values, self.stderr = values, stderr


@dataclass
class LoessFit:
    """A fitted LOESS model, mirroring what R's ``loess`` object provides."""

    x: np.ndarray
    y: np.ndarray
    span: float
    degree: int
    family: str
    surface: str
    #: Row labels of the observations actually used, so callers can scatter
    #: results back into the original frame the way R's named predict() lets
    #: them (``d2$..fit[as.numeric(names(fit))] <- fit``).
    used_index: np.ndarray
    _model: object

    @property
    def fitted(self) -> np.ndarray:
        return self.predict()

    def predict(self, newdata=None, se: bool = False):
        """R's ``predict.loess``.

        With ``newdata=None`` returns the fitted values for the observations
        used in the fit. ``se=True`` returns ``(fit, se_fit)``.
        """
        if newdata is None:
            xs = self.x
        else:
            xs = _as_predictor_matrix(newdata)
            if xs.shape[1] != self.x.shape[1]:
                raise ValueError(
                    f"newdata has {xs.shape[1]} predictor(s), model has {self.x.shape[1]}"
                )
        flat = xs[:, 0] if xs.shape[1] == 1 else xs
        if isinstance(self._model, _DirectModel) and newdata is None:
            flat = None
        out = self._model.predict(flat, stderror=se)
        values = np.asarray(out.values, dtype="float64")
        if se:
            return values, np.asarray(out.stderr, dtype="float64")
        return values

    def scatter(self, values: np.ndarray, n: int) -> np.ndarray:
        """Place ``values`` back at their original row positions, NA elsewhere.

        Reproduces the idiom DTEval uses after every LOESS fit, e.g.
        ``d2$..fit <- NA; d2$..fit[as.numeric(names(temp$fit))] <- temp$fit``
        (``deseason.tube.R:166``), which is how dropped NA rows stay NA.
        """
        out = np.full(n, np.nan)
        out[self.used_index] = values
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
    surface: str = "interpolate",
    normalize: bool = True,
    cell: float = 0.2,
) -> LoessFit:
    """Fit LOESS with R's defaults.

    Defaults match ``stats::loess``: ``span = 0.75``, ``degree = 2``,
    ``family = "gaussian"``, ``surface = "interpolate"``, ``cell = 0.2``.

    NA handling follows R's ``na.action = na.omit``: rows with a missing
    predictor or response are dropped before fitting, and
    :meth:`LoessFit.scatter` puts results back in the right places.
    """
    from skmisc.loess import loess as _skloess

    xs = _as_predictor_matrix(x)
    ys = pd.Series(y).to_numpy(dtype="float64")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError(f"x has {xs.shape[0]} rows, y has {ys.shape[0]}")

    if xs.shape[1] > 1 and surface != "direct":
        raise MultivariateLoessUnavailable(
            f"LOESS with {xs.shape[1]} predictors and surface={surface!r} is not "
            "available: scikit-misc 0.5.2's multivariate fit does not reproduce R "
            "(see MultivariateLoessUnavailable), and the interpolating surface "
            "needs R's ehg128 kd-tree, which is not ported. surface='direct' is "
            "implemented natively in dteval._loess_direct."
        )

    keep = ~(np.isnan(ys) | np.any(np.isnan(xs), axis=1))
    used_index = np.flatnonzero(keep)
    xs_fit, ys_fit = xs[keep], ys[keep]

    if xs.shape[1] > 1:
        return LoessFit(
            x=xs_fit, y=ys_fit, span=span, degree=degree, family=family,
            surface=surface, used_index=used_index,
            _model=_DirectModel(xs_fit, ys_fit, span, degree, normalize),
        )

    model = _skloess(
        xs_fit[:, 0],
        ys_fit,
        span=span,
        degree=degree,
        family=family,
        normalize=normalize,
        surface=surface,
        cell=cell,
    )
    model.fit()

    return LoessFit(
        x=xs_fit,
        y=ys_fit,
        span=span,
        degree=degree,
        family=family,
        surface=surface,
        used_index=used_index,
        _model=model,
    )

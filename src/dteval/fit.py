"""Fit models to diffusion tube data -- port of ``R/fit.tube.R``.

``fit_tube_model`` fits one model per group and returns the supplied data with
predictions attached, or predictions over a regular grid of the input ranges --
the latter being how ``tubePlot(plot.type = "surface")`` builds its raster.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from dteval.calc import calc_tube_stat
from dteval.gam import fit_tensor_gam
from dteval.handlers import DTEvalError, check_tube_data
from dteval.loess import r_loess
from dteval.rcompat.stats import r_mean
from dteval.tagging import tag_tube_required

__all__ = ["fit_tube_model", "fit_tube_model_gam", "fit_tube_model_loess"]


def fit_tube_model_loess(
    data: pd.DataFrame,
    tube: str,
    inputs: list[str],
    new_data: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``fitTubeModel_loess``.

    R builds ``[tube] ~ x1 * x2 * ...`` and fits with ``surface = "direct"``.
    In a LOESS formula the ``*`` simply names the predictors -- there is no
    interaction expansion -- so this is a direct-surface fit on ``inputs``.
    """
    fit = r_loess(data[inputs], data[tube], surface="direct")

    target = data if new_data is None else new_data
    out = target.copy()
    values, stderr = fit.predict(target[inputs], se=True)

    # R writes into pre-allocated NA columns using the names predict() carries,
    # so rows dropped for missing values stay NA.
    out["..pred"] = np.nan
    out["..pred.se"] = np.nan
    keep = ~np.isnan(np.asarray(target[inputs], dtype="float64")).any(axis=1)
    out.loc[out.index[keep], "..pred"] = values[keep]
    out.loc[out.index[keep], "..pred.se"] = stderr[keep]
    return out


def fit_tube_model_gam(
    data: pd.DataFrame,
    tube: str,
    inputs: list[str],
    new_data: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``fitTubeModel_gam``.

    R builds ``[tube] ~ te(x1, x2, ...)`` and fits with ``mgcv::gam``. This is
    the one backend that is close rather than equal to R -- see
    :mod:`dteval.gam` for what is reproduced and docs/parity.md for the
    measured deviation.
    """
    x = data[inputs].to_numpy(dtype="float64")
    fit = fit_tensor_gam(x, data[tube].to_numpy(dtype="float64"))

    target = data if new_data is None else new_data
    out = target.copy()
    grid = target[inputs].to_numpy(dtype="float64")
    values, stderr = fit.predict(grid, se=True)

    # As in the loess backend, rows with a missing input stay NA.
    out["..pred"] = np.nan
    out["..pred.se"] = np.nan
    keep = ~np.isnan(grid).any(axis=1)
    out.loc[out.index[keep], "..pred"] = values[keep]
    out.loc[out.index[keep], "..pred.se"] = stderr[keep]
    return out


def fit_tube_model(
    data: pd.DataFrame,
    tube: str = ".value",
    inputs: list[str] | str | None = None,
    by: list[str] | str | None = None,
    model: Callable | None = None,
    simplify: bool = False,
    min_count: float = -1,
    min_prop: float = -1,
    new_data: str | pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``fitTubeModel``.

    One model per ``by`` group. ``new_data='input.ranges'`` predicts over a
    regular grid spanning each input's range instead of over the data.
    ``simplify`` averages the tube value at each distinct input combination
    first, which is what makes a spatial fit tractable.
    """
    inputs = [] if inputs is None else ([inputs] if isinstance(inputs, str) else list(inputs))
    by_cols = [] if by is None else ([by] if isinstance(by, str) else list(by))

    d2 = tag_tube_required(data, required=[tube, *inputs, *by_cols], **kwargs)
    d2 = check_tube_data(d2, tube, if_err="stop<<fitTubeModel>>tube")
    if inputs:
        d2 = check_tube_data(d2, inputs, if_err="stop<<fitTubeModel>>inputs")
    if by_cols:
        d2 = check_tube_data(d2, by_cols, if_err="stop<<fitTubeModel>>by")

    if model is None:
        model = fit_tube_model_loess

    if simplify:
        d2 = calc_tube_stat(
            d2, tube, by=[*inputs, *by_cols], stat=lambda x: {"smooth": r_mean(x, na_rm=True)}
        )
        # R strips the stat suffix with gsub(".smooth", "", names(d2)), so
        # ".value.smooth" comes back as ".value".
        d2 = d2.rename(
            columns={c: c[: -len(".smooth")] for c in d2.columns if c.endswith(".smooth")}
        )

    d2 = d2.copy()
    if not by_cols:
        d2["..dummy"] = "default"
        by_cols = ["..dummy"]

    if len(by_cols) == 1:
        index = d2[by_cols[0]].astype(object)
    else:
        from dteval.rcompat.coerce import as_character_series

        index = None
        for col in by_cols:
            part = as_character_series(d2[col]).fillna("NA")
            index = part if index is None else index + "*" + part
    d2["..index"] = index

    grid = _new_data_grid(new_data, d2, inputs, kwargs)

    results = []
    for key in pd.unique(d2["..index"]):
        chunk = d2[d2["..index"] == key]
        values = chunk[tube].to_numpy(dtype="float64")
        present = int((~np.isnan(values)).sum())
        if min_count > 0 and present < min_count:
            continue
        if min_prop > 0 and present / len(values) < min_prop:
            continue

        try:
            out = model(chunk, tube, inputs, grid)
        except Exception:  # noqa: BLE001 -- R wraps this in try() and drops the group
            continue
        if out is None:
            continue

        out = out.rename(columns={"..pred": f"{tube}.pred"})
        for col in [*by_cols, "..index"]:
            out[col] = chunk[col].iloc[0]
        results.append(out)

    if not results:
        raise DTEvalError("[fitTubeModel] no viable models...")

    ans = pd.concat(results, ignore_index=True)
    ans = ans[[c for c in ans.columns if c != "..index"]]

    # Assigning a scalar per group demotes a factor to plain strings, and
    # factor level order is part of the output.
    from dteval.calc import _restore_dtype

    for col in by_cols:
        if col in ans.columns and col in d2.columns:
            ans[col] = _restore_dtype(ans[col], d2[col])
    return ans.rename(
        columns={"..pred": f"{tube}.pred", "..pred.se": f"{tube}.pred.se"}
    )


def _new_data_grid(new_data, d2: pd.DataFrame, inputs: list[str], kwargs: dict):
    """Build the prediction grid for ``new.data='input.ranges'``.

    ``expand.grid`` varies the first column fastest, which fixes the row order
    of the resulting surface.
    """
    if new_data is None:
        return None
    if isinstance(new_data, pd.DataFrame):
        return new_data
    if new_data != "input.ranges":
        import warnings

        warnings.warn(
            "[fitTubeModel] Sorry, did not understand input.range; ignoring...",
            stacklevel=2,
        )
        return None

    resolution = int(kwargs.get("grid.resolution", 100))
    borders = kwargs.get("grid.borders")
    axes = []
    for name in inputs:
        values = d2[name].to_numpy(dtype="float64")
        lo, hi = np.nanmin(values), np.nanmax(values)
        pad = (hi - lo) * borders if borders is not None else 0.0
        axes.append(np.linspace(lo - pad, hi + pad, resolution))

    mesh = np.meshgrid(*axes, indexing="ij")
    # expand.grid's first factor varies fastest, so reverse before flattening.
    flat = [m.ravel(order="F") for m in mesh]
    return pd.DataFrame(dict(zip(inputs, flat, strict=True)))

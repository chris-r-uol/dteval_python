"""Cluster diffusion tube sites -- port of ``R/cluster.tube.R``.

Groups sampling locations by how their time-series behave: by level, by
correlation, by normalised profile shape, or by trend direction.

Clustering algorithm
--------------------
R uses ``cluster::clara``, which is PAM applied to random *subsamples* for
speed on large data. Its result therefore depends on R's RNG stream, and
reproducing it would mean porting R's Mersenne-Twister and ``clara``'s sampling
scheme.

We run **PAM directly on the full data** instead. There are at most a few
hundred sites, so the cost clara exists to avoid does not arise, and the result
is both deterministic and the answer clara is approximating. Cluster *labels*
are arbitrary in either case; see docs/parity.md for how the parity suite
compares them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dteval.handlers import DTEvalError, check_tube_data
from dteval.rcompat.merge import r_merge
from dteval.tagging import tag_tube_required

__all__ = ["cluster_tube_data", "pam"]

METHODS = (1, 2, 3, 4, 5, 6)


def pam(x: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Partitioning Around Medoids -- deterministic, on the full data.

    The classic BUILD + SWAP formulation. Returns a 0-based cluster index per
    row. Ties are broken by the lowest index, so the result does not depend on
    iteration order.
    """
    n = len(x)
    if k >= n:
        return np.arange(n)

    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))

    # BUILD: first the most central point, then greedily whichever most
    # reduces the total distance to the nearest medoid.
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    while len(medoids) < k:
        nearest = dist[:, medoids].min(axis=1)
        gains = np.array(
            [
                -np.inf if j in medoids else float(np.minimum(nearest, dist[:, j]).sum())
                for j in range(n)
            ]
        )
        gains[np.isneginf(gains)] = np.inf
        medoids.append(int(np.argmin(gains)))

    # SWAP: accept the single best improving swap until none improves.
    best = float(dist[:, medoids].min(axis=1).sum())
    for _ in range(max_iter):
        improved = False
        for mi in range(k):
            for cand in range(n):
                if cand in medoids:
                    continue
                trial = list(medoids)
                trial[mi] = cand
                cost = float(dist[:, trial].min(axis=1).sum())
                if cost < best - 1e-12:
                    best, medoids, improved = cost, trial, True
        if not improved:
            break

    return np.argmin(dist[:, medoids], axis=1)


def cluster_tube_data(
    data: pd.DataFrame,
    tube: str = ".value",
    by: str | list[str] = "site",
    clusters: int = 2,
    method: int = 2,
    rename: str = ".cluster",
    output: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Port of ``clusterTubeData``.

    Methods (``cluster.tube.R:170``):

    1. cluster on the values themselves
    2. cluster on between-site correlation (default)
    3. cluster on min-max normalised profiles
    4. cluster on centred and scaled profiles
    5. cluster on profiles divided by their mean
    6. cluster on each site's correlation with time, i.e. trend direction
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    if method not in METHODS:
        raise DTEvalError(
            f"[clusterTubeData] Unknown method, maybe try one of: "
            f"{','.join(str(m) for m in METHODS)}"
        )

    data = tag_tube_required(data, required=[tube, *by_cols, ".date"], **kwargs)
    d2 = check_tube_data(data, tube, if_err="stop<<clusterTubeData>>tube")
    d2 = check_tube_data(d2, by_cols, if_err="stop<<clusterTubeData>>by")

    # Long -> wide: one row per date (plus any extra `by` terms), one column
    # per site, averaging duplicates.
    index = [".date", *by_cols[1:]]
    wide = d2.pivot_table(
        index=index, columns=by_cols[0], values=tube, aggfunc="mean", dropna=False
    )
    wide = wide.sort_index()

    if output == "common.data":
        return wide.reset_index()

    features, names = _features(wide, method)
    if output == "data":
        return pd.DataFrame(features, index=names)

    labels = pam(features, clusters) + 1

    assignment = pd.DataFrame({by_cols[0]: names, rename: labels})
    assignment[rename] = pd.Categorical(
        assignment[rename].astype(str), categories=[str(i + 1) for i in range(clusters)]
    )
    assignment[by_cols[0]] = assignment[by_cols[0]].astype(data[by_cols[0]].dtype)

    out = data[[c for c in data.columns if c != rename]]
    return r_merge(out, assignment, by=[by_cols[0]])


def _features(wide: pd.DataFrame, method: int) -> tuple[np.ndarray, list[str]]:
    """Build the matrix each method clusters, and the site names for its rows."""
    names = [str(c) for c in wide.columns]
    values = wide.to_numpy(dtype="float64")  # dates x sites

    if method == 2:
        # Correlation between sites, as a distance. R then clusters the *rows*
        # of that matrix as feature vectors rather than treating it as a
        # dist object.
        corr = pd.DataFrame(values, columns=names).corr(min_periods=1).to_numpy()
        dist = 1 - corr
        dist[np.isnan(dist)] = 1
        keep = ~np.all(dist == 1, axis=1)
        return dist[keep][:, keep], [n for n, k in zip(names, keep, strict=True) if k]

    if method == 6:
        # Correlation with time, reversed so a falling series scores positive.
        days = np.arange(len(values), dtype="float64")
        days = days.max() - days
        cols = []
        for j in range(values.shape[1]):
            col = values[:, j]
            ok = ~np.isnan(col)
            cols.append(
                np.corrcoef(col[ok], days[ok])[0, 1] if ok.sum() > 1 else np.nan
            )
        feat = 1 - np.asarray(cols)
        keep = ~np.isnan(feat)
        return feat[keep].reshape(-1, 1), [n for n, k in zip(names, keep, strict=True) if k]

    per_site = values.T  # sites x dates
    if method == 3:
        lo = np.nanmin(per_site, axis=1, keepdims=True)
        hi = np.nanmax(per_site, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_site = (per_site - lo) / (hi - lo)
    elif method == 4:
        mean = np.nanmean(per_site, axis=1, keepdims=True)
        sd = np.nanstd(per_site, axis=1, ddof=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_site = (per_site - mean) / sd
    elif method == 5:
        mean = np.nanmean(per_site, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_site = per_site / mean

    keep = ~np.all(np.isnan(per_site), axis=1)
    per_site = per_site[keep]
    # PAM needs finite coordinates; R's clara has correct.d for this.
    per_site = np.nan_to_num(per_site, nan=0.0, posinf=0.0, neginf=0.0)
    return per_site, [n for n, k in zip(names, keep, strict=True) if k]

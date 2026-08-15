"""Clustering: what the port reproduces, and where it deliberately differs.

``clusterTubeData`` calls ``cluster::clara`` -- PAM applied to random
subsamples for speed, so its answer depends on R's RNG stream. The port runs
PAM on the full data instead.

These tests separate the two questions: does our PAM match R's PAM (it must),
and how far does R's own clara sit from R's own PAM (context for the
documented parity gap).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import dteval as dte
from dteval.cluster import pam

FIXTURE = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "_rcompat.json.gz"


@pytest.fixture(scope="module")
def ref():
    if not FIXTURE.exists():
        pytest.skip("run Rscript parity/generate_rcompat.R")
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)["cluster"]


def _matrix(ref):
    dim = [int(v) for v in ref["dim"]]
    return np.array([float(v) for v in ref["features"]]).reshape(dim, order="F")


def _same_partition(a, b) -> bool:
    """Cluster labels are arbitrary; compare the partition they induce."""
    a, b = np.asarray(a), np.asarray(b)
    return all(len(set(b[a == c])) == 1 for c in np.unique(a)) and all(
        len(set(a[b == c])) == 1 for c in np.unique(b)
    )


def test_feature_matrix_matches_r(ref):
    """The matrix handed to the clusterer must match R's exactly."""
    got = dte.cluster_tube_data(
        dte.datasets.dt_brd(), tube=".value", by=".location", clusters=2,
        method=2, output="data",
    )
    assert np.nanmax(np.abs(got.to_numpy() - _matrix(ref))) < 1e-12


def test_our_pam_reproduces_rs_pam(ref):
    """Our PAM must give R's pam() partition -- this is the tight check."""
    got = pam(_matrix(ref), 2) + 1
    assert _same_partition(got, ref["pam"]), "partition differs from R's pam()"


def test_our_pam_matches_rs_pam_objective(ref):
    x = _matrix(ref)
    labels = pam(x, 2) + 1
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    cost = sum(
        dist[np.ix_(idx, idx)].sum(axis=0).min()
        for idx in (np.flatnonzero(labels == c) for c in np.unique(labels))
    ) / len(labels)
    assert cost == pytest.approx(float(ref["pam_objective"][0]), rel=1e-6)


def test_clara_is_a_worse_approximation_than_pam(ref):
    """Context for the documented gap: R's clara is not R's pam.

    clara samples 44 of the 157 sites by default, so it lands on a different --
    and by its own objective, worse -- partition than exact PAM. That is the
    whole of the divergence between this port and clusterTubeData.
    """
    clara_obj = float(ref["clara_objective"][0])
    pam_obj = float(ref["pam_objective"][0])
    assert pam_obj <= clara_obj, "PAM should never be worse than clara"
    assert not _same_partition(ref["clara"], ref["pam"]), (
        "expected clara and pam to disagree on this data"
    )

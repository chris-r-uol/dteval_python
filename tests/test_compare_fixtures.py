"""The fixture-vs-fixture gate CI runs after regenerating from live R.

A gate that cannot fail is worse than no gate, so these check both directions:
a last-ulp difference must pass (that is the whole reason it is not a byte
diff) and anything that changes the answer must fail.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest
from parity.compare import DEFAULT_TOL, Report
from parity.compare_fixtures import compare_directories, compare_nodes

FIXTURES = Path(__file__).resolve().parents[1] / "parity" / "fixtures"
CASE = "calc__calcTubeStat.by.year.json.gz"


def _report(exp, got) -> Report:
    rep = Report("test")
    compare_nodes(exp, got, "$", rep, DEFAULT_TOL)
    return rep


def _numeric_node(values: list[str]) -> dict:
    return {"rtype": "numeric", "values": values}


def test_identical_trees_agree():
    node = {"a": _numeric_node(["1.5", "2.5"]), "b": "text"}
    assert _report(node, json.loads(json.dumps(node))).ok


def test_a_last_ulp_difference_passes():
    """The reason this is not `git diff --exit-code`: R accumulates in long
    double where the build has it, so group means differ in the last bit
    between platforms."""
    value = 26.486934701610355
    exp = _numeric_node(["%.17g" % value])
    got = _numeric_node(["%.17g" % math.nextafter(value, math.inf)])
    assert _report(exp, got).ok


def test_a_meaningful_numeric_difference_fails():
    value = 26.486934701610355
    exp = _numeric_node(["%.17g" % value])
    got = _numeric_node(["%.17g" % (value * (1 + 1e-3))])
    assert not _report(exp, got).ok


@pytest.mark.parametrize(
    ("exp", "got"),
    [
        # a dropped element
        ({"cols": [_numeric_node(["1"]), _numeric_node(["2"])]}, {"cols": [_numeric_node(["1"])]}),
        # a renamed key
        ({"value": _numeric_node(["1"])}, {"values": _numeric_node(["1"])}),
        # a changed type
        ({"a": _numeric_node(["1"])}, {"a": {"rtype": "integer", "values": ["1"]}}),
        # a changed string
        ({"a": "left"}, {"a": "right"}),
    ],
)
def test_structural_differences_fail(exp, got):
    assert not _report(exp, got).ok


def test_na_and_nan_stay_distinct():
    """R's NA_real_ and NaN are different values and must not compare equal."""
    exp = _numeric_node(["NA"])
    got = _numeric_node(["NaN"])
    assert not _report(exp, got).ok


def test_a_directory_compared_with_itself_agrees():
    if not (FIXTURES / CASE).exists():
        pytest.skip("run Rscript parity/generate.R")
    assert compare_directories(FIXTURES, FIXTURES, DEFAULT_TOL) == []


def test_a_missing_fixture_is_reported(tmp_path):
    if not (FIXTURES / CASE).exists():
        pytest.skip("run Rscript parity/generate.R")
    partial = tmp_path / "partial"
    partial.mkdir()
    with gzip.open(partial / "only_one.json.gz", "wt") as fh:
        json.dump({"value": _numeric_node(["1"])}, fh)

    reports = compare_directories(FIXTURES, partial, DEFAULT_TOL)
    messages = " ".join(str(d) for rep in reports for d in rep.diffs)
    assert "was not regenerated" in messages
    assert "new and not committed" in messages


def test_provenance_files_are_not_compared(tmp_path):
    """_lock.json and _shims.json record the generating machine, so they
    legitimately differ; tests/test_lock.py gates the parts that must not."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    with gzip.open(a / "_lock.json", "wt") as fh:
        json.dump({"packages": {"sf": "1.1.2"}}, fh)
    with gzip.open(b / "_lock.json", "wt") as fh:
        json.dump({"packages": {"sf": "9.9.9"}}, fh)

    assert compare_directories(a, b, DEFAULT_TOL) == []

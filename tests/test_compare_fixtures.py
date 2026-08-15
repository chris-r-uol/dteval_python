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
from parity.compare_fixtures import (
    compare_directories,
    compare_nodes,
    load_cases,
    numeric_strings_agree,
)

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
    reports, _ = compare_directories(FIXTURES, FIXTURES, DEFAULT_TOL, load_cases())
    assert reports == []


def test_a_missing_fixture_is_reported(tmp_path):
    if not (FIXTURES / CASE).exists():
        pytest.skip("run Rscript parity/generate.R")
    partial = tmp_path / "partial"
    partial.mkdir()
    with gzip.open(partial / "only_one.json.gz", "wt") as fh:
        json.dump({"value": _numeric_node(["1"])}, fh)

    reports, _ = compare_directories(FIXTURES, partial, DEFAULT_TOL)
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

    assert compare_directories(a, b, DEFAULT_TOL) == ([], [])


def test_bare_numeric_strings_use_the_tolerance():
    """Some _rcompat blocks are plain %.17g character vectors, so they arrive
    as JSON strings and must still tolerate a last-ulp difference."""
    assert numeric_strings_agree("23.498466666666662", "23.498466666666666", DEFAULT_TOL)
    assert not numeric_strings_agree("23.4984", "23.5", DEFAULT_TOL)


def test_bare_non_numeric_strings_stay_exact():
    assert numeric_strings_agree("Bradford", "Bradford", DEFAULT_TOL) is None
    assert numeric_strings_agree("NA", "NaN", DEFAULT_TOL) is False
    assert numeric_strings_agree("NA", "NA", DEFAULT_TOL) is True


def test_non_gating_cases_are_skipped_not_compared():
    """deseason and cluster are documented approximations of R's own
    approximation; the fixture gate must agree with the manifest about that."""
    cases = load_cases()
    non_gating = [n for n, c in cases.items() if c.get("gating", True) is False]
    assert "deseason__method1.location.json.gz" in non_gating


def _numeric_list_report(exp, got, tol=DEFAULT_TOL) -> Report:
    rep = Report("test")
    compare_nodes(exp, got, "$", rep, tol)
    return rep


def test_a_near_zero_difference_is_noise_not_information():
    """R's cor() returns exactly 1 for a perfect correlation on one platform
    and 1 - eps/2 on another, so 1 - cor is 0 against 1.1e-16. A purely
    relative tolerance puts those infinitely far apart."""
    exp = ["2", "1.5", "1.1102230246251565e-16"]
    got = ["2", "1.5", "0"]
    assert _numeric_list_report(exp, got).ok


def test_the_noise_floor_stays_tighter_than_the_consuming_test():
    """test_cluster.py accepts 1e-12 absolute on this matrix, so the gate must
    reject anything at that size or the gate is the weaker of the two."""
    exp = ["2", "1.5", "1e-12"]
    got = ["2", "1.5", "0"]
    assert not _numeric_list_report(exp, got).ok


def test_the_noise_floor_scales_with_the_array():
    """Eight ulps of the array's own maximum, not a fixed constant."""
    small = _numeric_list_report(["1e-6", "1e-22"], ["1e-6", "0"])
    assert small.ok, "1e-22 is noise beside 1e-6"
    assert not _numeric_list_report(["1e-6", "1e-12"], ["1e-6", "0"]).ok


def test_a_real_difference_in_a_numeric_list_still_fails():
    assert not _numeric_list_report(["1.0", "2.0"], ["1.0", "2.5"]).ok


def test_a_list_of_non_numeric_strings_stays_exact():
    assert not _numeric_list_report(["Bradford", "Leeds"], ["Bradford", "York"]).ok

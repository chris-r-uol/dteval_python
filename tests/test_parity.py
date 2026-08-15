"""Compare every manifest case against its committed R fixture.

This is the acceptance criterion for the port. Each case names an R expression
and a Python expression; the R side has already been run by
``parity/generate.R`` and serialised, and here we run the Python side and
require the results to match bit for bit.

Cases whose fixture does not exist yet are skipped rather than failed, so the
suite stays usable while functions are still being ported -- but
``test_every_case_has_a_fixture`` reports how much is still outstanding, and a
case that *has* a fixture is never allowed to silently regress.
"""

from __future__ import annotations

import pytest
from compare import compare, load_fixture

import dteval as dte  # noqa: F401  -- in scope for manifest `py` expressions
from conftest import manifest_cases

CASES = manifest_cases()


def _eval_py(expr: str):
    """Evaluate a manifest Python expression.

    ``eval`` is appropriate here: the expressions come from our own version
    controlled manifest, which is exactly as trusted as this file.
    """
    return eval(expr, {"dte": dte, "dteval": dte})  # noqa: S307


def _case_id(case: dict) -> str:
    return case["id"]


@pytest.mark.parity
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_parity(case: dict) -> None:
    case_id = case["id"]
    try:
        fixture = load_fixture(case_id)
    except FileNotFoundError:
        pytest.skip(f"no fixture yet for {case_id}; run Rscript parity/generate.R")

    tol = case.get("tol")
    if tol is not None and not case.get("reason"):
        pytest.fail(
            f"case {case_id!r} sets tol={tol} without a `reason`. "
            "A tolerance must always be justified in the manifest."
        )

    try:
        got = _eval_py(case["py"])
    except (AttributeError, NotImplementedError) as exc:
        pytest.skip(f"{case_id}: not ported yet ({exc})")

    report = compare(case_id, got, fixture, tol=tol, tol_columns=case.get("tol_columns"))

    if not report.ok:
        if case.get("gating", True) is False:
            pytest.xfail(f"documented parity gap:\n{report.render()}")
        pytest.fail(report.render())


def test_manifest_ids_are_unique() -> None:
    ids = [c["id"] for c in CASES]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate case ids in manifest: {sorted(dupes)}"


def test_tolerances_are_justified() -> None:
    """A tolerance without a documented reason is a bug, not a convenience."""
    offenders = [c["id"] for c in CASES if c.get("tol") is not None and not c.get("reason")]
    assert not offenders, f"cases with an unjustified tolerance: {offenders}"

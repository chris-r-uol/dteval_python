"""Compare two fixture directories under the project's own parity contract.

Used by CI: regenerate the fixtures from live R, then check them against the
committed ones. The obvious way to do that is ``git diff --exit-code``, and
that is wrong -- it asserts *byte* identity of 17-significant-digit doubles,
which additionally requires both machines to agree on floating-point
accumulation. R sums in ``LDOUBLE``, so a build with
``capabilities("long.double") == TRUE`` (Linux x86_64) and one without
(conda-forge macOS arm64) differ in the last ulp of every group mean. Neither
is wrong, and docs/parity.md already states the gold standard is a *specific*
R build.

So the gate is the contract the project actually makes, applied between the two
fixture sets: **structure exactly, numbers to 1e-6**. That is a stronger check
than byte equality in the ways that matter -- a changed column, a lost factor
level, a reordered row or a shifted value all fail -- and it does not fail for
a last-digit difference that no diffusion tube could resolve.

Usage:  python parity/compare_fixtures.py <committed-dir> <regenerated-dir>
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

# Importable both as a script (`python parity/compare_fixtures.py`) and as
# `parity.compare_fixtures` under pytest, so put this directory on the path
# before reaching for its sibling.
sys.path.insert(0, str(Path(__file__).parent))

from compare import DEFAULT_TOL, Report, compare_series  # noqa: E402

from dteval.rcompat.rserial import load_column  # noqa: E402

MANIFEST = Path(__file__).parent / "manifest.yaml"


def fixture_name(case_id: str) -> str:
    """The filename generate.R writes a case to."""
    return f"{case_id.replace('/', '__')}.json.gz"


def load_cases() -> dict[str, dict]:
    """Manifest cases keyed by fixture filename.

    The fixture gate has to honour the same per-case settings the Python-vs-R
    suite does. A case marked `gating: false` is documented as not held to the
    contract -- deseason and cluster, where R's own answer is an approximation
    -- so holding its *fixture* to the contract instead would be incoherent.
    """
    import yaml

    manifest = yaml.safe_load(MANIFEST.read_text())
    return {fixture_name(case["id"]): case for case in manifest["cases"]}


def tolerance_for(node: dict, case: dict | None, default: float) -> float:
    """The tolerance for one column, honouring the case's tol / tol_columns."""
    if case is None:
        return default
    tol = case.get("tol", default)
    columns = case.get("tol_columns")
    name = node.get("name")
    if columns and name:
        if isinstance(columns, dict):
            return columns.get(name, tol)
        if name in columns:
            return tol
    return tol


#: Provenance, not fixtures: R package versions and which shims were needed
#: legitimately differ between machines. _lock.json's upstream SHA and source
#: hash are the parts that must hold, and tests/test_lock.py gates those.
PROVENANCE = {"_lock.json", "_shims.json"}


def compare_nodes(
    exp: Any, got: Any, path: str, rep: Report, tol: float, case: dict | None = None
) -> None:
    """Walk two serialised trees in parallel.

    The serialiser emits a small closed set of node shapes, so one recursive
    walk covers data frames, lists, matrices, ggplot specs and leaflet calls
    alike. Any dict carrying an ``rtype`` is a vector and is compared with the
    same numeric semantics the main parity suite uses; everything else must
    match exactly.
    """
    if isinstance(exp, dict) and isinstance(got, dict):
        if "rtype" in exp and "rtype" in got:
            compare_vectors(exp, got, path, rep, tolerance_for(exp, case, tol))
            return
        if exp.keys() != got.keys():
            only_exp = sorted(exp.keys() - got.keys())
            only_got = sorted(got.keys() - exp.keys())
            rep.add(path, f"keys differ: missing {only_exp}, unexpected {only_got}")
            return
        for key in exp:
            compare_nodes(exp[key], got[key], f"{path}.{key}", rep, tol, case)
        return

    if isinstance(exp, list) and isinstance(got, list):
        if len(exp) != len(got):
            rep.add(path, f"length {len(got)}, expected {len(exp)}")
            return
        if compare_numeric_list(exp, got, path, rep, tol):
            return
        for i, (a, b) in enumerate(zip(exp, got, strict=True)):
            compare_nodes(a, b, f"{path}[{i}]", rep, tol, case)
        return

    if isinstance(exp, str) and isinstance(got, str):
        verdict = numeric_strings_agree(exp, got, tol)
        if verdict is not None:
            if not verdict:
                rep.add(path, f"expected {exp!r}, got {got!r}")
            return

    if exp != got:
        rep.add(path, f"expected {exp!r}, got {got!r}")


#: A difference below this many ulps of an array's own scale is floating-point
#: noise, not information. R's `cor` returns exactly 1 for a perfectly
#: correlated pair on one platform and 1 - eps/2 on another, so `1 - cor` is 0
#: against 1.1e-16 -- infinitely far apart under a purely *relative* tolerance,
#: and identical for every purpose. Eight ulps leaves room for a two-pass
#: correlation to accumulate rounding while staying far tighter than the tests
#: that consume these blocks (test_cluster.py asks for 1e-12 absolute on a
#: matrix whose largest element is 2; eight ulps of 2 is 1.8e-15).
NOISE_ULPS = 8


def compare_numeric_list(
    exp: list, got: list, path: str, rep: Report, tol: float
) -> bool:
    """Compare two lists of serialised doubles. Returns False if they are not.

    Scale matters here: these are bare %.17g character vectors, so unlike a
    serialised column they carry no dtype, and a purely relative comparison
    breaks down for the elements near zero.
    """
    try:
        exp_values = [float(v) for v in exp]
        got_values = [float(v) for v in got]
    except (TypeError, ValueError):
        return False
    if any(v in SPECIAL_TOKENS for v in exp) or any(v in SPECIAL_TOKENS for v in got):
        return False

    finite = [abs(v) for v in exp_values + got_values if math.isfinite(v)]
    floor = max(finite, default=0.0) * NOISE_ULPS * 2.0**-52

    for i, (a, b) in enumerate(zip(exp_values, got_values, strict=True)):
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            if exp[i] != got[i]:
                rep.add(f"{path}[{i}]", f"expected {exp[i]!r}, got {got[i]!r}")
            continue
        if abs(a - b) <= max(tol * max(abs(a), abs(b)), floor):
            continue
        rep.add(f"{path}[{i}]", f"expected {exp[i]!r}, got {got[i]!r}")
    return True


def numeric_strings_agree(exp: str, got: str, tol: float) -> bool | None:
    """Compare two bare strings as serialised doubles, or return None.

    Some blocks in _rcompat.json.gz are plain %.17g character vectors rather
    than serialised columns, so they arrive as JSON strings and would otherwise
    be compared as text -- which fails on the last-ulp differences this gate
    exists to tolerate.

    Only bare strings reach here. Anything that is *data* arrives inside an
    `rtype: character` node and is compared exactly by compare_vectors, so
    widening these cannot loosen a real string comparison.
    """
    if exp in SPECIAL_TOKENS or got in SPECIAL_TOKENS:
        return exp == got
    try:
        a, b = float(exp), float(got)
    except ValueError:
        return None
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return exp == got
    return math.isclose(a, b, rel_tol=tol, abs_tol=0.0)


#: Tokens the serialiser writes instead of a number. They must be compared as
#: text: NA_real_ and NaN are different values in R, and load_column collapses
#: both to float64 nan because that is all pandas has.
SPECIAL_TOKENS = frozenset({"NA", "NaN", "Inf", "-Inf"})

#: Derived from `values` and documented as informational, so not compared.
DERIVED_KEYS = frozenset({"values", "rtype", "iso"})


def compare_vectors(exp: dict, got: dict, path: str, rep: Report, tol: float) -> None:
    """One serialised vector against another."""
    if exp["rtype"] != got["rtype"]:
        rep.add(path, f"rtype {got['rtype']!r}, expected {exp['rtype']!r}")
        return

    # Factor levels and their order, character NA positions, POSIXct tzone --
    # all part of the answer, none of them reachable through the loaded Series.
    for key in sorted((set(exp) | set(got)) - DERIVED_KEYS):
        if exp.get(key) != got.get(key):
            rep.add(f"{path}.{key}", f"expected {exp.get(key)!r}, got {got.get(key)!r}")
            return

    if exp["rtype"] == "numeric":
        exp_special = {i: v for i, v in enumerate(exp["values"]) if v in SPECIAL_TOKENS}
        got_special = {i: v for i, v in enumerate(got["values"]) if v in SPECIAL_TOKENS}
        if exp_special != got_special:
            where = sorted(set(exp_special) ^ set(got_special)) or [
                i for i in exp_special if exp_special[i] != got_special.get(i)
            ]
            i = where[0]
            rep.add(
                path,
                f"row {i}: expected {exp_special.get(i, 'a number')}, "
                f"got {got_special.get(i, 'a number')}",
            )
            return

    compare_series(load_column(exp), load_column(got), path, rep, tol)


def compare_directories(
    committed: Path, regenerated: Path, tol: float, cases: dict[str, dict] | None = None
) -> tuple[list[Report], list[str]]:
    cases = {} if cases is None else cases
    reports: list[Report] = []
    skipped: list[str] = []
    expected_files = {p.name for p in committed.glob("*.json.gz")} - PROVENANCE
    actual_files = {p.name for p in regenerated.glob("*.json.gz")} - PROVENANCE

    for name in sorted(expected_files - actual_files):
        rep = Report(name)
        rep.add("$", "fixture was not regenerated")
        reports.append(rep)
    for name in sorted(actual_files - expected_files):
        rep = Report(name)
        rep.add("$", "fixture is new and not committed")
        reports.append(rep)

    for name in sorted(expected_files & actual_files):
        case = cases.get(name)
        if case is not None and case.get("gating", True) is False:
            skipped.append(name)
            continue
        rep = Report(name)
        with gzip.open(committed / name, "rt") as fh:
            exp = json.load(fh)
        with gzip.open(regenerated / name, "rt") as fh:
            got = json.load(fh)
        compare_nodes(exp, got, "$", rep, tol, case)
        if not rep.ok:
            reports.append(rep)
    return reports, skipped


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    committed, regenerated = Path(argv[1]), Path(argv[2])
    reports, skipped = compare_directories(
        committed, regenerated, DEFAULT_TOL, load_cases()
    )
    for name in skipped:
        print(f"skipped (gating: false in the manifest): {name}")

    if not reports:
        print(
            f"all fixtures agree between {committed} and {regenerated} "
            f"(structure exact, numbers to {DEFAULT_TOL:g})"
        )
        return 0

    print(f"::error::{len(reports)} fixture(s) differ beyond the parity contract")
    for rep in reports:
        print(f"\n=== {rep.case_id} ===")
        print(rep.render(limit=10))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

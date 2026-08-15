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
import sys
from pathlib import Path
from typing import Any

# Importable both as a script (`python parity/compare_fixtures.py`) and as
# `parity.compare_fixtures` under pytest, so put this directory on the path
# before reaching for its sibling.
sys.path.insert(0, str(Path(__file__).parent))

from compare import DEFAULT_TOL, Report, compare_series  # noqa: E402

from dteval.rcompat.rserial import load_column  # noqa: E402

#: Provenance, not fixtures: R package versions and which shims were needed
#: legitimately differ between machines. _lock.json's upstream SHA and source
#: hash are the parts that must hold, and tests/test_lock.py gates those.
PROVENANCE = {"_lock.json", "_shims.json"}


def compare_nodes(exp: Any, got: Any, path: str, rep: Report, tol: float) -> None:
    """Walk two serialised trees in parallel.

    The serialiser emits a small closed set of node shapes, so one recursive
    walk covers data frames, lists, matrices, ggplot specs and leaflet calls
    alike. Any dict carrying an ``rtype`` is a vector and is compared with the
    same numeric semantics the main parity suite uses; everything else must
    match exactly.
    """
    if isinstance(exp, dict) and isinstance(got, dict):
        if "rtype" in exp and "rtype" in got:
            compare_vectors(exp, got, path, rep, tol)
            return
        if exp.keys() != got.keys():
            only_exp = sorted(exp.keys() - got.keys())
            only_got = sorted(got.keys() - exp.keys())
            rep.add(path, f"keys differ: missing {only_exp}, unexpected {only_got}")
            return
        for key in exp:
            compare_nodes(exp[key], got[key], f"{path}.{key}", rep, tol)
        return

    if isinstance(exp, list) and isinstance(got, list):
        if len(exp) != len(got):
            rep.add(path, f"length {len(got)}, expected {len(exp)}")
            return
        for i, (a, b) in enumerate(zip(exp, got, strict=True)):
            compare_nodes(a, b, f"{path}[{i}]", rep, tol)
        return

    if exp != got:
        rep.add(path, f"expected {exp!r}, got {got!r}")


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


def compare_directories(committed: Path, regenerated: Path, tol: float) -> list[Report]:
    reports = []
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
        rep = Report(name)
        with gzip.open(committed / name, "rt") as fh:
            exp = json.load(fh)
        with gzip.open(regenerated / name, "rt") as fh:
            got = json.load(fh)
        compare_nodes(exp, got, "$", rep, tol)
        if not rep.ok:
            reports.append(rep)
    return reports


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    committed, regenerated = Path(argv[1]), Path(argv[2])
    reports = compare_directories(committed, regenerated, DEFAULT_TOL)

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

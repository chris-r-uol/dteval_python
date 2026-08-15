"""The staleness gate on the committed fixtures.

``parity/fixtures/_lock.json`` records what produced the fixtures: the pinned
upstream commit, a hash of its R sources and data, the R version and the R
package versions. These tests assert the parts that must hold on any machine --
that the lock agrees with the manifest, that every declared case actually has a
fixture, and that the fixtures were not committed from a filtered run.

The R-source hash is only checkable where the pinned tree is present, so that
one skips without it. It is the check that catches the fixtures having been
generated against a different upstream than the one the manifest pins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "parity" / "fixtures" / "_lock.json"
MANIFEST = ROOT / "parity" / "manifest.yaml"
R_REFERENCE = Path(__file__).resolve().parents[1] / "r_reference"


@pytest.fixture(scope="module")
def lock() -> dict:
    if not LOCK.exists():
        pytest.skip("run Rscript parity/generate.R")
    return json.loads(LOCK.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def test_lock_pins_the_manifests_upstream(lock, manifest):
    assert lock["upstream_sha"] == manifest["meta"]["upstream_sha"]
    assert lock["upstream_repo"] == manifest["meta"]["upstream_repo"]


def test_lock_records_the_pinned_locale(lock, manifest):
    """Collation and time formatting are values here, not presentation."""
    assert lock["locale"] == manifest["meta"]["locale"]


def test_every_case_has_a_fixture(manifest):
    missing = [
        case["id"]
        for case in manifest["cases"]
        if not (ROOT / "parity" / "fixtures" / f"{case['id'].replace('/', '__')}.json.gz").exists()
    ]
    assert not missing, f"manifest cases with no fixture: {missing}"


def test_fixtures_are_not_from_a_filtered_run(lock, manifest):
    """`Rscript parity/generate.R <filter>` writes a lock saying so. Committing
    that would leave the other fixtures silently unaccounted for."""
    assert lock["n_cases"] == len(manifest["cases"]), (
        f"_lock.json says {lock['n_cases']} case(s) but the manifest declares "
        f"{len(manifest['cases'])}; regenerate with an unfiltered `make fixtures`"
    )


@pytest.mark.live_r
@pytest.mark.parametrize(("subdir", "key"), [("R", "r_source_hash"), ("data", "r_data_hash")])
def test_pinned_tree_still_hashes_to_the_lock(lock, subdir, key):
    """The fixtures cannot silently outlive the sources that produced them."""
    tree = R_REFERENCE / subdir
    if not tree.is_dir():
        pytest.skip("run `make r-reference` to clone the pinned upstream tree")
    assert _hash_tree(tree) == lock[key], (
        f"r_reference/{subdir} no longer hashes to _lock.json's {key}. Either the "
        "pinned tree moved or the fixtures are stale; regenerate and review."
    )


def _hash_tree(directory: Path) -> str:
    """Reimplementation of ``hash_tree`` in parity/generate.R.

    R builds one ``"<relative path> <md5>"`` line per file, sorted by full path
    under C collation, and md5s the result -- with the trailing newline
    ``writeLines`` adds.
    """
    paths = sorted(
        (p for p in directory.rglob("*") if p.is_file() and ".git" not in p.parts),
        key=lambda p: str(p).encode(),
    )
    lines = [
        f"{p.relative_to(directory).as_posix()} {hashlib.md5(p.read_bytes()).hexdigest()}"
        for p in paths
    ]
    return hashlib.md5(("\n".join(lines) + "\n").encode()).hexdigest()[:64]

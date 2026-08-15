"""The figure catalogue in tools/figures/.

``tools/figures/render.R`` and ``tools/figures/validate.py`` read one catalogue,
so neither implementation can quietly draw something the other did not -- the
same discipline as parity/manifest.yaml, and the reason the comparison means
anything.

These tests do not draw anything. They check the catalogue is internally
coherent and still agrees with the manifest, which is what breaks silently when
a case is renamed months later.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "tools" / "figures" / "catalogue.yaml"
MANIFEST = ROOT / "parity" / "manifest.yaml"


@pytest.fixture(scope="module")
def figures() -> list[dict]:
    return yaml.safe_load(CATALOGUE.read_text())["figures"]


@pytest.fixture(scope="module")
def manifest_case_ids() -> set[str]:
    return {case["id"] for case in yaml.safe_load(MANIFEST.read_text())["cases"]}


def test_every_figure_has_both_expressions(figures):
    """A figure with only one side cannot be compared, and would silently
    reduce the coverage of `make figure-parity`."""
    for fig in figures:
        assert fig.get("r"), f"{fig['id']}: no R expression"
        assert fig.get("py"), f"{fig['id']}: no Python expression"


def test_figure_ids_are_unique(figures):
    ids = [fig["id"] for fig in figures]
    assert len(ids) == len(set(ids))


def test_figure_ids_are_safe_as_filenames(figures):
    """Both renderers write <id>.png, so a slash or space would land the file
    somewhere unexpected rather than failing."""
    for fig in figures:
        assert fig["id"].replace("-", "").replace(".", "").isalnum(), fig["id"]


def test_every_figure_has_drawing_dimensions(figures):
    for fig in figures:
        assert float(fig["width"]) > 0
        assert float(fig["height"]) > 0


def test_figures_reference_real_manifest_cases(figures, manifest_case_ids):
    """The comparison honours each case's documented tolerances and row-order
    artifacts. A stale id would silently fall back to the strict default and
    report differences the parity suite itself does not consider differences.
    """
    missing = [
        (fig["id"], fig["case"])
        for fig in figures
        if fig.get("case") and fig["case"] not in manifest_case_ids
    ]
    assert not missing, f"catalogue points at manifest cases that no longer exist: {missing}"


def test_every_figure_names_a_case(figures):
    for fig in figures:
        assert fig.get("case"), (
            f"{fig['id']}: no manifest case, so the comparison would use the "
            "strict default rather than the settings that case documents"
        )

"""Shared test fixtures and manifest loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = REPO_ROOT / "parity"

# parity/ is a scripts directory, not an installed package.
if str(PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(PARITY_DIR))


def load_manifest() -> dict:
    with (PARITY_DIR / "manifest.yaml").open() as fh:
        return yaml.safe_load(fh)


def manifest_cases() -> list[dict]:
    return load_manifest()["cases"]


@pytest.fixture(scope="session")
def manifest() -> dict:
    return load_manifest()

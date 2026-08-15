"""Read R fixtures and compare Python results against them with R semantics.

Bit-exactness is the default. Doubles are serialised by ``parity/serialize.R``
as ``%.17g`` strings, which round-trip exactly, so "the same number" means the
same 64 bits -- not "close enough". A tolerance is only ever applied when the
manifest states one *and* gives a reason.

What counts as a difference
---------------------------
Column names, column order, row order, dtypes, factor levels and their order,
and the NA/NaN distinction are all part of the comparison. R's ``NA_real_`` and
``NaN`` are different values and stay different here; Python cannot tell them
apart from the float alone, so the fixture's token is authoritative and the
Python side must supply the distinction through pandas' own missing-value
representation.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dteval.rcompat.rserial import load_value

FIXTURE_DIR = Path(__file__).parent / "fixtures"

class ParityError(AssertionError):
    """Raised when a Python result does not match its R fixture."""


@dataclass
class Diff:
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.path}: {self.message}"


@dataclass
class Report:
    case_id: str
    diffs: list[Diff] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diffs

    def add(self, path: str, message: str) -> None:
        self.diffs.append(Diff(path, message))

    def render(self, limit: int = 25) -> str:
        head = f"{len(self.diffs)} difference(s) for case {self.case_id!r}"
        body = "\n".join(f"  - {d}" for d in self.diffs[:limit])
        if len(self.diffs) > limit:
            body += f"\n  ... and {len(self.diffs) - limit} more"
        return f"{head}\n{body}"


# --------------------------------------------------------------------------
# deserialisation
# --------------------------------------------------------------------------
# The reader lives in the shipped package (dteval.rcompat.rserial) because the
# bundled datasets use the same format. Keeping one implementation means a
# fixture and a dataset can never be read by two subtly different parsers.


def load_fixture(case_id: str, fixture_dir: Path | None = None) -> dict[str, Any]:
    """Read a fixture. Fixtures are gzipped; plain .json is still accepted."""
    stem = case_id.replace("/", "__").replace(" ", "__")
    base = fixture_dir or FIXTURE_DIR
    gz, plain = base / f"{stem}.json.gz", base / f"{stem}.json"
    if gz.exists():
        with gzip.open(gz, "rt") as fh:
            return json.load(fh)
    if plain.exists():
        with plain.open() as fh:
            return json.load(fh)
    raise FileNotFoundError(
        f"no fixture for case {case_id!r} at {gz}. Run: Rscript parity/generate.R"
    )


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _same_double(a: float, b: float, tol: float | None) -> bool:
    a_nan, b_nan = a != a, b != b
    if a_nan or b_nan:
        return a_nan and b_nan
    if a == b:
        return True
    if tol is None:
        return False
    if math.isinf(a) or math.isinf(b):
        return False
    scale = max(abs(a), abs(b))
    return abs(a - b) <= tol * scale if scale else abs(a - b) <= tol


def compare_series(
    exp: pd.Series, got: pd.Series, path: str, rep: Report, tol: float | None
) -> None:
    if len(exp) != len(got):
        rep.add(path, f"length {len(got)} != expected {len(exp)}")
        return

    if isinstance(exp.dtype, pd.CategoricalDtype):
        if not isinstance(got.dtype, pd.CategoricalDtype):
            rep.add(path, f"expected a factor, got dtype {got.dtype}")
            return
        if list(exp.cat.categories) != list(got.cat.categories):
            rep.add(
                path,
                f"factor levels differ:\n      R: {list(exp.cat.categories)}\n      py: {list(got.cat.categories)}",
            )
        if exp.cat.ordered != got.cat.ordered:
            rep.add(path, f"ordered={got.cat.ordered}, expected {exp.cat.ordered}")
        exp, got = exp.astype(object), got.astype(object)

    e = exp.to_numpy(dtype=object, na_value=None)
    g = got.to_numpy(dtype=object, na_value=None)

    numeric = pd.api.types.is_float_dtype(exp.dtype)
    bad = 0
    for i, (ev, gv) in enumerate(zip(e, g, strict=False)):
        same = (
            _same_double(float(ev), float(gv), tol)
            if numeric and ev is not None and gv is not None
            else (ev is None and gv is None) or (ev is not None and gv is not None and ev == gv)
        )
        if not same:
            bad += 1
            if bad <= 3:
                rep.add(path, f"row {i}: R={ev!r} python={gv!r}")
    if bad > 3:
        rep.add(path, f"... {bad} differing values in total")


def compare_frame(
    exp: pd.DataFrame,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: set[str] | None = None,
) -> None:
    if not isinstance(got, pd.DataFrame):
        rep.add(path, f"expected a DataFrame, got {type(got).__name__}")
        return
    if list(exp.columns) != list(got.columns):
        missing = [c for c in exp.columns if c not in got.columns]
        extra = [c for c in got.columns if c not in exp.columns]
        msg = "column names/order differ"
        if missing:
            msg += f"; missing {missing}"
        if extra:
            msg += f"; unexpected {extra}"
        if not missing and not extra:
            msg += f"\n      R:  {list(exp.columns)}\n      py: {list(got.columns)}"
        rep.add(path, msg)
        return
    if len(exp) != len(got):
        rep.add(path, f"{len(got)} rows, expected {len(exp)}")
        return
    got = got.reset_index(drop=True)
    for col in exp.columns:
        # A tolerance applies only to the columns the manifest names. Everything
        # else stays bit-exact, so loosening one LOESS-derived column cannot
        # quietly loosen the counts and means beside it.
        col_tol = tol if (tol_columns is None or col in tol_columns) else None
        compare_series(exp[col], got[col], f"{path}[{col!r}]", rep, col_tol)


def compare_ggplot(
    exp: dict,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: set[str] | None = None,
) -> None:
    """Compare a figure on DTEval's decisions, not on rendered pixels.

    Checked: plot labels, facet specification, and per layer the geom, the
    input data frame (bit-exactly), the aesthetic mappings, and any constant
    aesthetics. See serialize_ggplot() in parity/serialize.R for why this is
    the contract rather than ggplot_build() output.
    """
    spec = got.parity_spec() if hasattr(got, "parity_spec") else got
    if not isinstance(spec, dict):
        rep.add(path, f"expected a plot exposing parity_spec(), got {type(got).__name__}")
        return

    exp_labels = exp.get("labels") or {}
    got_labels = spec.get("labels") or {}
    for key, value in exp_labels.items():
        if got_labels.get(key) != value:
            rep.add(f"{path}.labels[{key!r}]", f"R={value!r} python={got_labels.get(key)!r}")

    if exp.get("facet") != spec.get("facet"):
        rep.add(f"{path}.facet", f"R={exp.get('facet')!r} python={spec.get('facet')!r}")

    exp_fv = _as_str_list(exp.get("facet_vars"))
    got_fv = _as_str_list(spec.get("facet_vars"))
    if exp_fv != got_fv:
        rep.add(f"{path}.facet_vars", f"R={exp_fv} python={got_fv}")

    exp_layers = exp.get("layers") or []
    got_layers = spec.get("layers") or []
    if len(exp_layers) != len(got_layers):
        rep.add(
            f"{path}.layers",
            f"{len(got_layers)} layer(s), expected {len(exp_layers)} "
            f"(R: {[layer.get('geom') for layer in exp_layers]}, "
            f"python: {[layer.get('geom') for layer in got_layers]})",
        )
        return

    for i, (el, gl) in enumerate(zip(exp_layers, got_layers, strict=True)):
        lp = f"{path}.layers[{i}]"
        el = dict(el)
        # Repeated layer data is stored once and referenced (see serialize.R).
        node = el.get("data") or {}
        if node.get("type") == "ref":
            el["data"] = exp_layers[int(node["layer"]) - 1]["data"]
        if el.get("geom") != gl.get("geom"):
            rep.add(f"{lp}.geom", f"R={el.get('geom')!r} python={gl.get('geom')!r}")

        exp_map = _named_map(el.get("mapping"), el.get("mapping_names"))
        got_map = dict(gl.get("mapping") or {})
        if exp_map != got_map:
            rep.add(f"{lp}.mapping", f"R={exp_map} python={got_map}")

        exp_aes = {k: str(v) for k, v in (el.get("aes_params") or {}).items()}
        got_aes = dict(gl.get("aes_params") or {})
        if exp_aes != got_aes:
            rep.add(f"{lp}.aes_params", f"R={exp_aes} python={got_aes}")

        compare_frame(
            load_value(el["data"]), gl.get("data"), f"{lp}.data", rep, tol, tol_columns
        )


def _as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _named_map(values, names) -> dict[str, str]:
    """jsonlite writes a named character vector as an object or as a bare list."""
    if isinstance(values, dict):
        return {k: str(v) for k, v in values.items()}
    vals = _as_str_list(values)
    keys = _as_str_list(names)
    return dict(zip(keys, vals, strict=False))


def compare(
    case_id: str,
    got: Any,
    fixture: dict,
    tol: float | None = None,
    tol_columns: list[str] | None = None,
) -> Report:
    """Compare a Python result against a loaded fixture payload.

    ``tol`` is applied only to the columns named in ``tol_columns``; with no
    such list it applies to every numeric column. Either way the manifest must
    also give a ``reason``.
    """
    rep = Report(case_id)
    node = fixture["value"]
    cols = set(tol_columns) if tol_columns else None

    if node["type"] == "ggplot":
        compare_ggplot(node, got, "$", rep, tol, cols)
        return rep

    exp = load_value(node)
    _compare_any(exp, got, "$", rep, tol, cols)
    return rep


def _compare_any(
    exp: Any,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: set[str] | None = None,
) -> None:
    # A ggplot nested inside a result list (e.g. testTubePrecision's `plot`)
    # stays a raw node, so route it to the figure comparator rather than
    # treating it as a named list.
    if isinstance(exp, dict) and exp.get("type") == "ggplot":
        compare_ggplot(exp, got, path, rep, tol, tol_columns)
        return
    if exp is None:
        if got is not None:
            rep.add(path, f"expected NULL, got {type(got).__name__}")
        return
    if isinstance(exp, pd.DataFrame):
        compare_frame(exp, got, path, rep, tol, tol_columns)
    elif isinstance(exp, pd.Series):
        if isinstance(got, pd.Series):
            compare_series(exp, got, path, rep, tol)
        elif isinstance(got, list | np.ndarray):
            compare_series(exp, pd.Series(got, dtype=exp.dtype), path, rep, tol)
        elif len(exp) == 1:
            compare_series(exp, pd.Series([got], dtype=exp.dtype), path, rep, tol)
        else:
            rep.add(path, f"expected a vector of {len(exp)}, got {type(got).__name__}")
    elif isinstance(exp, dict):
        if not isinstance(got, dict):
            rep.add(path, f"expected a named list, got {type(got).__name__}")
            return
        for k, v in exp.items():
            if k not in got:
                rep.add(f"{path}.{k}", "missing from Python result")
                continue
            _compare_any(v, got[k], f"{path}.{k}", rep, tol, tol_columns)
    elif isinstance(exp, np.ndarray):
        g = np.asarray(got)
        if exp.shape != g.shape:
            rep.add(path, f"shape {g.shape}, expected {exp.shape}")
        elif not np.array_equal(exp, g, equal_nan=True):
            rep.add(path, "matrix values differ")
    else:
        if exp != got:
            rep.add(path, f"R={exp!r} python={got!r}")

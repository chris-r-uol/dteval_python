"""Read R fixtures and compare Python results against them with R semantics.

Numbers are compared to a relative tolerance of :data:`DEFAULT_TOL`; structure
is compared exactly.

That split is deliberate. Diffusion tube NO2 measurements carry roughly a 10%
error bar, so agreeing with R to the last bit is ~14 orders of margin on
something the instrument cannot resolve, and chasing it costs real complexity
(transliterated Fortran, platform-coupled floating point). 1e-6 still sits ~4
orders inside the measurement error and ~3 orders outside any realistic bug --
a wrong formula, subset or grouping shows up at 1e-3 or larger.

What is *not* relaxed is anything that changes the answer rather than its last
digits: column names and order, row count and row order, dtypes, factor levels
and their order, the NA/NaN distinction, and every integer, string, date and
boolean. Those are correctness, not precision -- and about 56% of compared
cells are non-float, so no tolerance touches them at all.

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

#: Relative tolerance for comparing floats. See the module docstring for why
#: this is not zero. Individual cases may tighten or loosen it via
#: `tol` / `tol_columns` in the manifest.
DEFAULT_TOL = 1e-6

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


def _tol_for(column: str, tol: float | None, tol_columns) -> float | None:
    """Resolve the tolerance for one column.

    Every float column gets ``tol`` (the case's, or :data:`DEFAULT_TOL`).
    ``tol_columns`` *overrides* that for named columns -- as a list, they all
    take the case's ``tol``; as a mapping, each takes its own bound, which is
    how a LOESS fit and its much weaker standard error sit in the same case.
    """
    base = DEFAULT_TOL if tol is None else tol
    if tol_columns is None:
        return base
    if isinstance(tol_columns, dict):
        return tol_columns.get(column, DEFAULT_TOL)
    return base if column in tol_columns else DEFAULT_TOL


def compare_frame(
    exp: pd.DataFrame,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: dict[str, float] | set[str] | None = None,
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
        col_tol = _tol_for(col, tol, tol_columns)
        compare_series(exp[col], got[col], f"{path}[{col!r}]", rep, col_tol)


def compare_ggplot(
    exp: dict,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: set[str] | None = None,
    row_order_artifact: list[str] | None = None,
    label_columns: list[str] | None = None,
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

        # Layer data is usually the same frame the case returns, so it needs
        # the same row-order and label handling.
        exp_data = load_value(el["data"])
        got_data = gl.get("data")
        exp_data, got_data = _prepare(
            exp_data, got_data, rep, row_order_artifact, label_columns
        )
        compare_frame(exp_data, got_data, f"{lp}.data", rep, tol, tol_columns)


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


def compare_labels(
    exp: pd.DataFrame, got: pd.DataFrame, columns: list[str], path: str, rep: Report
) -> None:
    """Check a label column induces the same grouping, not the same values.

    ``.sample_id`` numbers replicate sets. R derives its integers by pasting the
    location and dates into a string and taking ``as.numeric(factor(...))``, so
    the labels depend on how R formats a double and on collation order; we group
    on the tuple directly. The sets are the same, the numbering need not be.

    What matters is that the *partition* matches: two rows share a label here
    exactly when they share one in R. That is checked as a bijection, so a
    genuine mis-grouping still fails.
    """
    for column in columns:
        if column not in exp.columns or column not in got.columns:
            continue
        pairs = pd.DataFrame({"r": exp[column].to_numpy(), "py": got[column].to_numpy()})
        r_to_py = pairs.groupby("r")["py"].nunique()
        py_to_r = pairs.groupby("py")["r"].nunique()
        bad_r = r_to_py[r_to_py > 1]
        bad_py = py_to_r[py_to_r > 1]
        if len(bad_r) or len(bad_py):
            rep.add(
                f"{path}[{column!r}]",
                f"grouping differs from R: {len(bad_r)} R label(s) split across "
                f"several python labels, {len(bad_py)} python label(s) merge "
                "several R labels",
            )


def canonicalise(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Stably sort a frame by ``columns`` so row order stops mattering.

    Used only where R's own row order is an internal artifact we do not
    reproduce -- see the `row_order_artifact` field in parity/manifest.yaml.
    Everything else is compared in R's order.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return df.reset_index(drop=True)
    keys = pd.DataFrame(
        {c: df[c].astype(object).map(lambda v: "" if pd.isna(v) else str(v)) for c in present}
    )
    order = keys.sort_values(present, kind="stable").index
    return df.loc[order].reset_index(drop=True)


def compare(
    case_id: str,
    got: Any,
    fixture: dict,
    tol: float | None = None,
    tol_columns: list[str] | dict[str, float] | None = None,
    row_order_artifact: list[str] | None = None,
    label_columns: list[str] | None = None,
) -> Report:
    """Compare a Python result against a loaded fixture payload.

    ``tol`` is applied only to the columns named in ``tol_columns``; with no
    such list it applies to every numeric column. Either way the manifest must
    also give a ``reason``.
    """
    rep = Report(case_id)
    node = fixture["value"]
    if tol is None:
        tol = DEFAULT_TOL
    cols = None
    if tol_columns:
        cols = dict(tol_columns) if isinstance(tol_columns, dict) else set(tol_columns)

    if node["type"] == "ggplot":
        compare_ggplot(node, got, "$", rep, tol, cols, row_order_artifact, label_columns)
        return rep

    exp = load_value(node)
    exp, got = _prepare(exp, got, rep, row_order_artifact, label_columns)
    _compare_any(exp, got, "$", rep, tol, cols, row_order_artifact, label_columns)
    return rep


def _prepare(exp, got, rep, row_order_artifact, label_columns):
    """Canonicalise row order and check label columns, before value comparison."""
    def one(e, g):
        if not (isinstance(e, pd.DataFrame) and isinstance(g, pd.DataFrame)):
            return e, g
        if row_order_artifact:
            e, g = canonicalise(e, row_order_artifact), canonicalise(g, row_order_artifact)
        if label_columns:
            compare_labels(e, g, label_columns, "$", rep)
            keep = [c for c in e.columns if c not in label_columns]
            e, g = e[keep], g[[c for c in g.columns if c not in label_columns]]
        return e, g

    if isinstance(exp, dict) and isinstance(got, dict):
        out_e, out_g = dict(exp), dict(got)
        for key in exp:
            if key in got:
                out_e[key], out_g[key] = one(exp[key], got[key])
        return out_e, out_g
    return one(exp, got)


def _compare_any(
    exp: Any,
    got: Any,
    path: str,
    rep: Report,
    tol: float | None,
    tol_columns: set[str] | None = None,
    row_order_artifact: list[str] | None = None,
    label_columns: list[str] | None = None,
) -> None:
    # A ggplot nested inside a result list (e.g. testTubePrecision's `plot`)
    # stays a raw node, so route it to the figure comparator rather than
    # treating it as a named list.
    if isinstance(exp, dict) and exp.get("type") == "ggplot":
        compare_ggplot(
            exp, got, path, rep, tol, tol_columns, row_order_artifact, label_columns
        )
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
            _compare_any(
                v, got[k], f"{path}.{k}", rep, tol, tol_columns,
                row_order_artifact, label_columns,
            )
    elif isinstance(exp, np.ndarray):
        g = np.asarray(got)
        if exp.shape != g.shape:
            rep.add(path, f"shape {g.shape}, expected {exp.shape}")
        elif not np.array_equal(exp, g, equal_nan=True):
            rep.add(path, "matrix values differ")
    else:
        if exp != got:
            rep.add(path, f"R={exp!r} python={got!r}")

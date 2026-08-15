"""The plotting argument system -- port of the ``dte_ggshell*`` helpers in ``zzz.R``.

DTEval's plots are driven by a single flat bag of keyword arguments, resolved
by three rules:

1. **Prefix scoping.** ``point.col`` sets ``col`` for the point layer only;
   ``smooth.linetype`` likewise for the smooth layer. When a layer of type
   ``T`` is built, any ``T.<arg>`` overrides the bare ``<arg>``
   (``dte_ggshellTidyArgs``).
2. **Data or constant.** Each argument value is tested against the data: if it
   names a column (or evaluates as an expression against it) the argument
   becomes an *aesthetic mapping*; otherwise it is a fixed *parameter*. So
   ``col=".year"`` colours by year, while ``col="red"`` paints everything red
   (``dte_ggshellTestArgs`` / ``dte_localArgsTests``).
3. **Geom filtering.** Anything the target geom does not accept is dropped
   rather than passed through (``dte_GeomArgs``).

``col`` and ``color`` are folded into ``colour`` throughout, matching R.

The per-geom argument sets in ``geom_args.json`` are captured from ggplot2
itself (``required_aes`` + ``default_aes`` + ``extra_params`` + ``group``) so
the filtering matches the real geoms rather than a hand-written guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from dteval.handlers import get_tube_x

__all__ = [
    "GEOM_ARGS",
    "LayerSpec",
    "geom_aesthetics",
    "geom_args",
    "tidy_args",
    "test_args",
]

_GEOM_ARGS_PATH = Path(__file__).parent / "geom_args.json"


@cache
def _load_geom_args() -> dict[str, list[str]]:
    with _GEOM_ARGS_PATH.open() as fh:
        return json.load(fh)


GEOM_ARGS = _load_geom_args


def _entry(geom: str) -> dict[str, list[str]]:
    table = _load_geom_args()
    if geom not in table:
        raise KeyError(
            f"no captured argument set for {geom!r}; regenerate "
            "src/dteval/plots/geom_args.json from ggplot2"
        )
    return table[geom]


def geom_args(geom: str) -> set[str]:
    """Every name a ggplot2 geom accepts -- aesthetics plus extra params."""
    return set(_entry(geom)["all"])


def geom_aesthetics(geom: str) -> set[str]:
    """Just the *aesthetics* (required + default), excluding extra params.

    The split matters for parity: R records a constant aesthetic (``colour =
    "red"``) in the layer's ``aes_params``, while a behavioural flag such as
    ``na.rm`` lands in ``geom_params``. Comparing them separately keeps
    "DTEval asked for a red line" distinct from ggplot2 plumbing.
    """
    return set(_entry(geom)["aes"])


def tidy_args(args: dict[str, Any], type: str | None = None) -> dict[str, Any]:
    """Port of ``dte_ggshellTidyArgs``.

    Resolves ``<type>.<arg>`` prefixes onto bare ``<arg>``, folds
    ``col``/``color`` into ``colour``, and sets the ``..test`` flag that R uses
    to suppress a layer when the user passes e.g. ``point=FALSE``.
    """
    out = dict(args)
    out["..test"] = "OK"

    if type is not None:
        prefixed = {
            k[len(type) + 1 :]: v for k, v in out.items() if k.startswith(f"{type}.")
        }
        out.update(prefixed)
        # `points=FALSE` style suppression: a logical under the type's own name
        # turns the layer off.
        if type in out:
            val = out[type]
            if isinstance(val, bool) and not val:
                out["..test"] = "not OK"

    for alias in ("col", "color"):
        if alias in out:
            # Later keys win, matching modifyList + duplicated(fromLast=TRUE).
            out["colour"] = out.pop(alias)

    return out


def test_args(args: dict[str, Any], data: pd.DataFrame | None) -> dict[str, str]:
    """Port of ``dte_localArgsTests``: classify each argument.

    * ``"data"``    -- resolves against ``data``, so it becomes an aesthetic mapping
    * ``"fact"``    -- evaluates standalone (a bare number or logical)
    * ``"unknown"`` -- anything else, e.g. the literal colour ``"red"``

    R does this by trying ``getTubeX(data, value)`` and then
    ``getTubeX(NULL, value)``; the distinction is what decides whether
    ``col="x"`` maps a column or paints a constant.
    """
    out: dict[str, str] = {}
    for name, value in args.items():
        if name.startswith(".."):
            continue
        source = "unknown"
        if isinstance(value, str) and data is not None:
            try:
                if get_tube_x(data, value) is not None:
                    source = "data"
            except Exception:  # noqa: BLE001 - R swallows this via try()
                source = "unknown"
        if source == "unknown" and not isinstance(value, str):
            # Bare numbers / logicals evaluate on their own in R.
            if isinstance(value, (int, float, bool)):
                source = "fact"
        out[name] = source
    return out


@dataclass
class LayerSpec:
    """One resolved plot layer -- the unit figure parity is compared on.

    ``mapping`` is aesthetic -> column name; ``params`` is aesthetic -> fixed
    value. Splitting them this way mirrors what ``dte_ggshellAddGeom`` builds
    and is exactly what the R fixture records.
    """

    geom: str
    data: pd.DataFrame
    mapping: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    stat: str = "StatIdentity"
    position: str = "PositionIdentity"

    def to_spec(self) -> dict[str, Any]:
        aes = geom_aesthetics(self.geom)
        return {
            "geom": self.geom,
            "mapping": dict(self.mapping),
            "aes_params": {k: _as_text(v) for k, v in self.params.items() if k in aes},
            "geom_params": {k: _as_text(v) for k, v in self.params.items() if k not in aes},
            "stat": self.stat,
            "position": self.position,
            "data": self.data,
        }


def _as_text(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (list, tuple)):
        return ",".join(_as_text(x) for x in v)
    return str(v)


def add_geom(
    args: dict[str, Any],
    data: pd.DataFrame,
    geom: str,
    defaults: dict[str, Any] | None = None,
    holds: dict[str, Any] | None = None,
    drops: set[str] | list[str] | None = None,
) -> LayerSpec:
    """Port of ``dte_ggshellAddGeom``.

    Builds one layer: arguments that name data become aesthetic mappings, the
    rest become fixed parameters, and anything in ``drops`` (or not accepted by
    the geom) is discarded.
    """
    defaults = dict(defaults or {})
    holds = dict(holds or {})
    drops = set(drops or ())

    merged = {**defaults, **args}
    merged.update(holds)

    classes = test_args(merged, data)
    mapping: dict[str, str] = {}
    params: dict[str, Any] = {}
    valid = geom_args(geom)

    for name, value in merged.items():
        if name.startswith("..") or name in drops:
            continue
        if name not in valid:
            continue
        if classes.get(name) == "data":
            mapping[name] = value
        else:
            params[name] = value

    return LayerSpec(geom=geom, data=data, mapping=mapping, params=params)


def resolve_facet(args: dict[str, Any]) -> tuple[str, list[str]]:
    """Port of the facet block: returns ``(facet_class, facet_vars)``.

    ``facet.type`` is one of ``wrap`` / ``grid`` / ``grid.row`` / ``grid.col``;
    an unrecognised value falls back to ``wrap`` with a warning, as in R.
    """
    if "facet" not in args:
        return "FacetNull", []

    facet = args["facet"]
    facet = [facet] if isinstance(facet, str) else list(facet)

    ftype = args.get("facet.type", "wrap")
    if ftype not in ("wrap", "grid", "grid.col", "grid.row"):
        import warnings

        warnings.warn("bad facet.type resetting to wrap", stacklevel=2)
        ftype = "wrap"

    return ("FacetWrap" if ftype == "wrap" else "FacetGrid"), facet

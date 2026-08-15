"""Render the catalogue with the Python port, and check it against R's.

Run ``tools/figures/render.R`` first: it draws the same catalogue with R DTEval
and dumps each plot's layer data. This then draws them with plotnine and
compares.

The images are for looking at; the comparison is what decides whether the two
implementations drew the same figure. It reuses ``parity/compare.py``, so the
rule is the project's own contract -- structure exactly, numbers to 1e-6 -- and
cannot drift from the gate CI runs.

Usage:  python tools/figures/validate.py [output-dir]
"""
from __future__ import annotations

import gzip
import json
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
# parity/compare.py is the comparator CI uses; reusing it is the point, so this
# check cannot drift from the gate. Note nothing in this directory may be named
# compare.py or it would shadow it on the path.
sys.path.insert(0, str(ROOT / "parity"))

from compare import DEFAULT_TOL, Report, compare_ggplot  # noqa: E402
from compare_fixtures import load_cases  # noqa: E402

import dteval as dte  # noqa: E402,F401

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "figures"


def main() -> int:
    figures = yaml.safe_load((Path(__file__).parent / "catalogue.yaml").read_text())["figures"]
    specs_path = OUT / "r" / "specs.json.gz"
    if not specs_path.exists():
        print(f"no R specs at {specs_path}\n"
              f"run:  FIG_OUT={OUT / 'r'} Rscript tools/figures/render.R", file=sys.stderr)
        return 2
    with gzip.open(specs_path, "rt") as fh:
        r_specs = json.load(fh)

    # The manifest already records which columns are row-order artifacts and
    # which tolerances a case is held to. Re-deriving that here would let this
    # check drift from the one CI runs -- and in practice makes it *stricter*
    # than the contract, reporting differences nobody would act on.
    cases = {c["id"]: c for c in load_cases().values()}

    py_dir = OUT / "py"
    py_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], 0
    for fig in figures:
        case = cases.get(fig.get("case"), {})
        plot = eval(fig["py"], {"dte": dte})  # noqa: S307 - our own catalogue
        plot.draw().save(py_dir / f"{fig['id']}.png", width=fig["width"],
                         height=fig["height"], dpi=130, verbose=False)
        rep = Report(fig["id"])
        compare_ggplot(
            r_specs[fig["id"]], plot, "$", rep,
            case.get("tol", DEFAULT_TOL),
            case.get("tol_columns"),
            case.get("row_order_artifact"),
            case.get("label_columns"),
        )
        spec = plot.parity_spec()
        cells = sum(
            len(layer["data"]) * len(layer["data"].columns) for layer in spec["layers"]
        )
        rows.append({
            "id": fig["id"],
            "layers": [layer["geom"] for layer in spec["layers"]],
            "rows": sum(len(layer["data"]) for layer in spec["layers"]),
            "cells": cells,
            "ok": rep.ok,
            "case": fig.get("case"),
            "settings": {
                k: case.get(k) for k in
                ("tol", "tol_columns", "row_order_artifact", "label_columns")
                if case.get(k)
            },
            "diffs": [str(d) for d in rep.diffs[:5]],
        })
        failures += 0 if rep.ok else 1
        mark = "OK " if rep.ok else "FAIL"
        print(f"  {mark} {fig['id']:<14} {cells:>7,} cells  "
              f"{'+'.join(layer['geom'].replace('Geom', '') for layer in spec['layers'])}")
        if not rep.ok:
            print(rep.render(limit=5))

    (OUT / "validation.json").write_text(json.dumps(rows, indent=2))
    total = sum(r["cells"] for r in rows)
    print(f"\n{len(rows) - failures}/{len(rows)} figures agree; "
          f"{total:,} layer-data cells compared at {DEFAULT_TOL:g}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

UPSTREAM_REPO := https://github.com/karlropkins/DTEval.git
UPSTREAM_SHA  := 08a6cbda6d5703ef564f076f5ae5aafc1d7ad6d6
R_REFERENCE   ?= r_reference
PY            ?= .venv/bin/python

.PHONY: figures figure-parity help r-reference r-shims fixtures datasets test parity lint clean

help:
	@echo "make r-reference  clone/checkout the pinned upstream DTEval tree"
	@echo "make r-shims      build minimal loa/AQEval/OpenStreetMap shims (no JDK)"
	@echo "make datasets     re-export bundled datasets from the R .rda files"
	@echo "make fixtures     regenerate all parity fixtures from live R"
	@echo "make figures      draw the Python-native figure set"
	@echo "make figure-parity  draw the catalogue in R and Python, compare layers"
	@echo "make test         run the full test suite"
	@echo "make parity       run only the parity comparisons"
	@echo "make lint         ruff check"

# The upstream package is pinned by SHA rather than vendored, so the fixtures
# provably come from a known tree and this repo stays free of GPL R sources.
r-reference:
	@if [ ! -d "$(R_REFERENCE)/.git" ]; then \
		git clone --quiet $(UPSTREAM_REPO) $(R_REFERENCE); \
	fi
	@cd $(R_REFERENCE) && git fetch --quiet origin && git checkout --quiet $(UPSTREAM_SHA)
	@echo "upstream DTEval at $(UPSTREAM_SHA)"

# Minimal loa/AQEval/OpenStreetMap shims so fixtures can be generated without
# a JDK.
# See tools/build_r_shims.R.
r-shims:
	Rscript tools/build_r_shims.R

datasets: r-reference
	Rscript tools/export_datasets.R

# generate.R first: it is what installs DTEval into the project-local .Rlib,
# and generate_rcompat.R needs the package loadable.
fixtures: r-reference r-shims
	Rscript parity/generate.R
	Rscript parity/generate_rcompat.R

# Python-native figures (matplotlib). Needs dteval[plots,basemap]; OSM tiles
# are cached under ~/.cache/dteval/tiles after the first run.
figures:
	python tools/figures/native.py

# Draw the shared catalogue with BOTH implementations and compare the layer
# data. Needs R as well; see tools/figures/catalogue.yaml.
figure-parity: r-reference r-shims
	Rscript tools/figures/render.R
	python tools/figures/validate.py

test:
	$(PY) -m pytest

parity:
	$(PY) -m pytest -m parity

lint:
	$(PY) -m ruff check src tests parity tools

clean:
	rm -rf parity/_out .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

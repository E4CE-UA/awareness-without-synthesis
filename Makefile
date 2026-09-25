# Awareness Without Synthesis — reproduction entry points.
#
#   make install     editable install with test extras
#   make test        run the test suite
#   make check       non-mutating integrity checks (labels + provenance)
#   make reproduce   full canonical reproduction from the bundled demo data
#   make provenance  regenerate provenance.json
#   make clean       remove generated artefacts
#
# `make reproduce` is the single command referenced by the manuscript. It
# verifies the demo cluster identifiers, byte-compiles every module, runs the
# test suite, recomputes the canonical CSC via the CLI, regenerates the
# fragmentation map and verifies the committed provenance record. It does not
# rewrite provenance.json, so a successful run leaves the tree clean; use
# `make provenance` after intentionally changing tracked content.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
export PYTHONPATH := $(CURDIR)

FIGURE := examples/fragmentation_map.png

.PHONY: all install test check check-labels provenance verify-provenance \
        diagnose figure reproduce clean

all: reproduce

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

check-labels:
	$(PYTHON) tools/fix_cluster_labels.py --check

verify-provenance:
	$(PYTHON) tools/make_provenance.py --check

check: check-labels verify-provenance

provenance:
	$(PYTHON) tools/make_provenance.py

diagnose:
	$(PYTHON) -m aws_align.cli diagnose

figure:
	$(PYTHON) -m aws_align.cli map --out $(FIGURE)

reproduce: check-labels
	@echo "== 1/5 compiling all modules =="
	$(PYTHON) -m compileall -q aws_align tools
	@echo "== 2/5 test suite =="
	$(PYTHON) -m pytest -q
	@echo "== 3/5 canonical CSC diagnostic =="
	$(PYTHON) -m aws_align.cli diagnose
	@echo "== 4/5 fragmentation map =="
	$(PYTHON) -m aws_align.cli map --out $(FIGURE)
	@echo "== 5/5 verifying the committed provenance record =="
	$(PYTHON) tools/make_provenance.py --check
	@echo
	@echo "reproduction complete: canonical CSC = 0.402 (see provenance.json)"

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

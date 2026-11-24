# Run `make help` for targets. Each project has its own requirements file.
PY ?= python3
PROJECTS = survival-ml-pbc polymer-monte-carlo heart-failure-ml

.PHONY: help install test quick all survival polymer heart clean

help:
	@echo "install  - pip install all requirements"
	@echo "test     - run every test suite"
	@echo "quick    - fast smoke runs (minutes), as in CI"
	@echo "all      - full runs that regenerate results/ (about 15-30 min)"

install:
	for p in $(PROJECTS); do $(PY) -m pip install -r $$p/requirements.txt; done

test:
	for p in $(PROJECTS); do (cd $$p && $(PY) -m pytest -q) || exit 1; done

quick:
	cd survival-ml-pbc && $(PY) src/run.py --quick
	cd polymer-monte-carlo && $(PY) scripts/run_free_chain.py --quick && $(PY) scripts/run_adsorption.py --quick
	cd heart-failure-ml && $(PY) src/run.py --quick

survival:
	cd survival-ml-pbc && $(PY) src/run.py
polymer:
	cd polymer-monte-carlo && $(PY) scripts/run_free_chain.py && $(PY) scripts/run_adsorption.py
heart:
	cd heart-failure-ml && $(PY) src/run.py
all: survival polymer heart

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

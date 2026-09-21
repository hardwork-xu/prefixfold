UV ?= .venv/bin/uv
PY ?= .venv/bin/python

.PHONY: install demo test check benchmark analyze build validate
install:
	$(UV) sync --frozen --group bench

demo:
	$(PY) -m prefixfold demo

test:
	$(PY) -m pytest -q

check:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .
	$(UV) run --frozen mypy src/prefixfold
	$(PY) scripts/check_docs.py

benchmark:
	$(PY) benchmark.py --config configs/benchmark.json --output results/benchmark.json

analyze:
	$(PY) scripts/analyze.py --input results/benchmark.json --output-dir results

build:
	$(PY) -m build
	$(PY) -m twine check dist/*

validate:
	$(PY) scripts/validate.py

.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

qa: ## fix style, sort imports, check types
	uv run --extra test black . --include '\.ipynb$$|\.py$$'
	uv run --extra test ruff check . --fix
	uv run --extra test ruff check --select I --fix .
	uv run --extra test ruff format .
	# type check, should reactivate later
	# uv run --extra test ty check .

MAKECMDGOALS ?= .	

test:  ## Run all the tests, but allow for arguments to be passed
	@echo "Running with arg: $(filter-out $@,$(MAKECMDGOALS))"
	pytest -v $(filter-out $@,$(MAKECMDGOALS))

pdb:  ## Run all the tests, but on failure, drop into the debugger
	@echo "Running with arg: $(filter-out $@,$(MAKECMDGOALS))"
	pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb $(filter-out $@,$(MAKECMDGOALS))

test-all: ## run tests on every Python version with uv
	uv run --python=3.12 --extra test pytest
	uv run --python=3.13 --extra test pytest

coverage: ## check code coverage quickly with the default Python
	coverage run --source segtraq -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

NOTEBOOK_SRCS := $(wildcard docs/notebooks/*.py)
NOTEBOOK_OUTS := $(patsubst docs/notebooks/%.py,docs/_build/notebooks/%.ipynb,$(NOTEBOOK_SRCS))

docs/_build/notebooks/%.ipynb: docs/notebooks/%.py
	mkdir -p docs/_build/notebooks
	uv run --extra docs jupytext --to ipynb -o $@ $<
	uv run --extra docs jupyter nbconvert --to notebook --execute --inplace $@

.PHONY: docs
docs: $(NOTEBOOK_OUTS) ## build docs, only re-executing notebooks whose .py source changed
	uv run --extra docs sphinx-build -b html docs docs/_build/html

deploy-docs: docs
	uv run ghp-import -n -p -f docs/_build/html

release: dist ## package and upload a release
	uv release -t $(UV_PUBLISH_TOKEN)

build: clean ## builds source and wheel package
	rm -rf build dist
	uv build
	ls -l build dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

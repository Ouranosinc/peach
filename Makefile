include .env
.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 lint/black
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
LOCALES := docs/locales

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-envs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove docs artifacts
	rm -f docs/apidoc/peach*.rst
	rm -f docs/apidoc/modules.rst
	rm -fr docs/locales/fr/LC_MESSAGES/*.mo
	$(MAKE) -C docs clean

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

clean-envs: ## remove merged envs
	rm -f environment-backend-full.yml environment-frontend-full.yml environment-full.yml environment-dev-full.yml

lint/flake8: ## check style with flake8
	ruff peach tests
	flake8 --config=.flake8 peach tests

lint/black: ## check style with black
	black --check peach tests
	blackdoc --check peach docs
	isort --check peach tests

lint: lint/flake8 lint/black ## check style

test: ## run tests quickly with the default Python
	python -m pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source peach -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
initialize-translations: clean-docs ## initialize translations, ignoring autodoc-generated files
	${MAKE} -C docs gettext
	sphinx-intl update -p docs/_build/gettext -d docs/locales -l fr

autodoc: clean-docs ## create sphinx-apidoc files:
	sphinx-apidoc -o docs/apidoc --private --module-first src/peach

linkcheck: autodoc ## run checks over all external links found throughout the documentation
	$(MAKE) -C docs linkcheck

docs: autodoc ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html BUILDDIR="_build/html/en"
ifneq ("$(wildcard $(LOCALES))","")
	${MAKE} -C docs gettext
	$(MAKE) -C docs html BUILDDIR="_build/html/fr" SPHINXOPTS="-D language='fr'"
endif
ifndef READTHEDOCS
	$(BROWSER) docs/_build/html/en/html/index.html
endif

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m flit build
	ls -l dist

release: dist ## package and upload a release
	python -m flit publish dist/*

install: clean ## install the package to the active Python's site-packages
	python -m flit install

dev: clean ## install the package to the active Python's site-packages
	python -m flit install --symlink

### DOCKER IMAGES ###
export MY_USERNAME := $(or $(MY_USERNAME),$(shell whoami))
export MY_UID := $(or $(MY_UID),$(shell id -u $(MY_USERNAME)))
export MY_GID := $(or $(MY_GID),$(shell id -g $(MY_USERNAME)))

build-images:
ifneq ("$(wildcard environment-backend-full.yml)","")
    # file exists.
else
	echo "Missing full.yml environments. Merging environments"
	$(MAKE) env-merge
endif
	$(MAKE) stop-images
	echo "Building images with MY_USERNAME=$${MY_USERNAME} MY_UID=$${MY_UID} MY_GID=$${MY_GID}"
	docker compose build frontend-dev backend-dev build-docs

run-images:
	echo "Starting portail-ing servers"
	docker compose up -d  build-docs frontend-dev backend-dev

follow-logs:
	docker compose logs -f --tail=50 frontend-dev backend-dev

run-bash-backend-dev:
	docker compose run --rm -it --entrypoint "/bin/bash" backend-dev

build-docs:
	docker compose up -d  build-docs & docker exec build-docs bash "/quarto-run/build.sh"

stop-images:
	docker compose down -v frontend-dev backend-dev build-docs

### CONDA ENVIRONMENTS ###
env: env-merge env-lock

env-merge: # merge conda environment files into one, for docker dev images.
	pip install conda-merge
	conda merge environment.yml environment-backend.yml > environment-backend-full.yml
	conda merge environment.yml environment-frontend.yml > environment-frontend-full.yml
	conda merge environment.yml environment-dev.yml > environment-dev-full.yml
	conda merge environment.yml environment-frontend.yml environment-backend.yml > environment-full.yml

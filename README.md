# PEACH - Probabilistic Ensemble Analysis of Climate Hazards - v0.1.0

| Versions | [![pypi](https://img.shields.io/pypi/v/peach.svg)](https://pypi.python.org/pypi/peach) [![versions](https://img.shields.io/pypi/pyversions/peach.svg)](https://pypi.python.org/pypi/peach) |
|---|---|
| Documentation and Support | [![docs](https://readthedocs.org/projects/peach/badge/?version=latest)]() |
| Open Source | [![license](https://img.shields.io/pypi/l/peach)](https://github.com/Ouranosinc/peach/blob/main/LICENSE) [![ossf](https://api.securityscorecards.dev/projects/github.com/Ouranosinc/peach/badge)](https://securityscorecards.dev/viewer/?uri=github.com/Ouranosinc/peach) [![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17187211.svg)](https://doi.org/10.5281/zenodo.17187211) |
| Coding Standards | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Ouranosinc/peach/main.svg)](https://results.pre-commit.ci/latest/github/Ouranosinc/peach/main) |
| Development Status | [![status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![build](https://github.com/Ouranosinc/peach/actions/workflows/main.yml/badge.svg)](https://github.com/Ouranosinc/peach/actions) [![Coverage Status](https://coveralls.io/repos/github/Ouranosinc/peach/badge.svg?branch=main)](https://coveralls.io/github/Ouranosinc/peach?branch=main) |

PEACH is an online calculation service and Python package that offers calculation for climate hazard likelihood.
It computes climate indicators over hundreds of different Earth System Model (ESM) simulations, then combine the results into a single probability density function, accounting for model performance and the relative likelihood of greenhouse gas (GHG) emission scenarios.
Peach can be used to deploy calculation services and graphical interface or as a traditional Python package to estimate climate hazard likelihood.

## Features
- Relies on more than 500 climate model simulations from CMIP6, bias-adjusted relative to weather station records across Canada;
- Applies weights to GHG scenarios and ESMs to provide a probabilistic estimate of hazards;
- Provides a computational backend with an OGCAPI-Processes interface for the calculation of climate indicator time series and the probabilistic analysis;
- Provides a web interface prototype to run the analysis;
- Peer-reviewed, see [Huard et al. (2026)](https://doi.org/10.1088/2515-7620/ae3a4d)

## Data availability

- The bias adjusted data is available on this [FRDR repository](https://www.frdr-dfdr.ca/repo/dataset/876e9380-63fc-4eaa-987b-aa16c3770941);
- The logic to prepare the sea level data is detailed in the [pre-workflow folder](pre_workflow_data/).

## Credits
This project was funded by Infrastructure Canada' Research and Knowledge Initiative and the Québec government. It is led by [Ouranos](https://www.ouranos.ca/fr) with the contribution of [Institut national de la recherche scientifique (INRS-ETE)](https://inrs.ca/en/inrs/research-centres/eau-terre-environnement-research-centre/), [CBCL](https://www.cbcl.ca/), and [ClimAtlantic](https://climatlantic.ca/).

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [Ouranosinc/cookiecutter-pypackage](https://github.com/Ouranosinc/cookiecutter-pypackage) project template.

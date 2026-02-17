# Documentation for PEACH

Welcome to the documentation for the Probabilistic Ensemble Analysis of Climate Hazards (PEACH) project.
The objective is to answer the question:

> How likely is it that a climate hazard will exceed a threshold in the future?

The method underling the software is described in [Huard et al. (2026)](https://doi.org/10.1088/2515-7620/ae3a4d).

The code in the repo includes:
- a data analysis module;
- a server handling data-intensive backend computations;
- a prototype graphical user interface.

This documentation is intended for

- developers wanting to deploy the services, and
- advanced users looking for programmatic access to the services.

We're always intersted in feedback and suggestions to improve the method and the software. Please submit your ideas as Github [issues](https://github.com/Ouranoinc/peach/issues).

```{note}
Most of the documentation is in english, but some parts are in french.
```

```{toctree}
:caption: For developers
:hidden:
:maxdepth: 1

installation
deployment
storage
contributing
changes
authors
```

```{toctree}
:caption: For users
:hidden:
:maxdepth: 1

usage
notebooks/introduction
notebooks/metho_figures
notebooks/Joint_Example
```

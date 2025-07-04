#!/usr/bin/env python

"""Tests for `portail_ing` package."""

import pathlib
import pkgutil

import numpy as np
import pytest
import xarray as xr
from scipy import stats

from portail_ing.risk import base

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: https://doc.pytest.org/en/latest/explanation/fixtures.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_package_metadata():
    """Test the package metadata."""
    project = pkgutil.get_loader("portail_ing").get_filename()

    metadata = pathlib.Path(project).resolve().parent.joinpath("__init__.py")

    with open(metadata) as f:
        contents = f.read()
        assert """Sarah-Claude Bourdeau-Goulet""" in contents
        assert '__email__ = "bourdeau-goulet.sarah-claude@ouranos.ca"' in contents
        assert '__version__ = "0.1.0"' in contents


def test_ds_ks():
    """Test ds_ks function in risk/base.py"""
    n1 = np.array([0, 0, 0, 0, 0])
    n2 = np.array([0, 0, 1, 1, 1])
    n3 = np.array([1, 1, 1, 1, 1])

    ds1 = xr.Dataset(
        data_vars={
            "var1": (["realization", "time"], np.array([n1, n3])),
            "var2": (["realization", "time"], np.array([n1, n2])),
        },
        coords={"time": np.arange(5), "realization": [0, 1]},
    )

    ds2 = xr.Dataset(
        data_vars={
            "var1": (["time"], n3),
            "var2": (["time"], n2),
        },
        coords={"time": np.arange(5)},
    )

    ds_ks = base.ds_ks(ds1, ds2)
    int13 = stats.ks_2samp(n1, n3)
    res13 = [
        int13.statistic,
        int13.pvalue,
        int13.statistic_location,
        int13.statistic_sign,
    ]
    int12 = stats.ks_2samp(n1, n2)
    res12 = [
        int12.statistic,
        int12.pvalue,
        int12.statistic_location,
        int12.statistic_sign,
    ]
    int33 = stats.ks_2samp(n3, n3)
    res33 = [
        int33.statistic,
        int33.pvalue,
        int33.statistic_location,
        int33.statistic_sign,
    ]

    np.testing.assert_array_equal(ds_ks.var1.sel(realization=0), res13)
    np.testing.assert_array_equal(ds_ks.var2.sel(realization=0), res12)
    np.testing.assert_array_equal(ds_ks.var1.sel(realization=1), res33)

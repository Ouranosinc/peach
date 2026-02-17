.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17187211.svg
   :target: https://doi.org/10.5281/zenodo.17187211

===================================================================
PEACH - Probabilistic Ensemble Analysis of Climate Hazards - v0.1.0
===================================================================

+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Versions                   | |pypi| |versions|                                                                                                                 |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Documentation and Support  | |docs|                                                                                                                            |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Open Source                | |license| |ossf| |zenodo|                                                                                                         |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Coding Standards           | |black| |ruff| |pre-commit|                                                                                                       |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                                                                                                      |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------------+



PEACH is an online calculations services and python package that offers calculation for climate hazard likelihood.
Peach can be used to deploy calculation services and graphical interface or as traditional python package to estimate climate hazard likelihood.

Features
--------
- relies on more than 500 biasadjusted climate model simulations from CMIP6, please consult the `FRDR repository <https://www.frdr-dfdr.ca/repo/dataset/876e9380-63fc-4eaa-987b-aa16c3770941>`_ and `Pre-Workflow folder <pre_workflow_data/>`_;
- applies weights to SSPs (experiment_id) and models (source_id) to provide a probabilistic estimate of the hazard;
- provides a computational backend with an OGCAPI-Processes interface;
- provides a web interface prototype to run the analysis;

Development Servers
^^^^^^^^^^^^^^^^^^^
In order to create development images, first copy `.env.example` to `.env` and adjust the settings as needed.

Run the following commands to create the docker images for the frontend and backend, start them, and follow the logs:

.. code-block:: console

    $ make build-images && make run-images && make follow-logs

Note that the frontend image will merge the environment.yml and environment-frontend.yml for its requirements. Similarly for the backend image with environment-backend.yml.

Both the backend and frontend have auto-reloading enabled in development mode, but it may take a few seconds for these to propagate.
Notably, due to a bug in Gunicorn, the backend server will completely restart on any file change in the ``/src/`` folder.

Production Servers
^^^^^^^^^^^^^^^^^^
For a production server, you can start the images as follows:

.. code-block:: console

    $ docker compose up -d --build frontend backend && docker compose logs -f --tail 10


The differences between the production and development images are as follows:

- Auto-reloading is turned off.
- The local pip module is installed in non-editable mode. (``pip install --no-deps .`` instead of ``pip install --no-deps -e .``).

Otherwise, the images remain identical, and, unless specified otherwise in your .env file, will use the same ports by default.

Documentation Webpages
^^^^^^^^^^^^^^^^^^^^^^

The documentation webpages are rebuilt entirely whenever the `build-docs` image is restarted. It may also be rebuilt with the following `make` command:

.. code-block:: console

    $ make build-docs

On the production PAVICS server, it is rebuilt on changes to files in `docs/factsheets` every 15 minutes.

S3-like storage
^^^^^^^^^^^^^^^
The backend will write to a S3-like storage if the BUCKET_URL and BUCKET_NAME environment variables are given. A credentials file ``credentials.json`` must also be passed in order to have the backend.

It expects some sort of read-write policy looking like this:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::portail-ing",
                "arn:aws:s3:::portail-ing/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::portail-ing/workspace/*"
            ]
        }
    ]
}
```


Credits
-------

This project was funded by Infrastructure Canada' Research and Knowledge Initiative and the Québec government. It is led by [Ouranos ](https://www.ouranos.ca/fr) with the contribution of Institut national de la recherche scientifique (INRS-ETE_), CBCL_ and ClimAtlantic_.

This package was created with Cookiecutter_ and the `Ouranosinc/cookiecutter-pypackage`_ project template.

.. _INRS-ETE: https://inrs.ca/en/inrs/research-centres/eau-terre-environnement-research-centre/
.. _CBCL: https://www.cbcl.ca/
.. _ClimAtlantic: https://climatlantic.ca/
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
.. _`Ouranosinc/cookiecutter-pypackage`: https://github.com/Ouranosinc/cookiecutter-pypackage


.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |build| image:: https://github.com/Ouranosinc/peach/actions/workflows/main.yml/badge.svg
        :target: https://github.com/Ouranosinc/peach/actions
        :alt: Build Status

.. |coveralls| image:: https://coveralls.io/repos/github/Ouranosinc/peach/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/peach
        :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/peach/badge/?version=latest
        :target: https://peach.readthedocs.io/en/latest/
        :alt: Documentation Status

.. |license| image:: https://img.shields.io/pypi/l/peach
        :target: https://github.com/Ouranosinc/peach/blob/main/LICENSE
        :alt: License

.. |ossf| image:: https://api.securityscorecards.dev/projects/github.com/Ouranosinc/peach/badge
        :target: https://securityscorecards.dev/viewer/?uri=github.com/Ouranosinc/peach
        :alt: OpenSSF Scorecard

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/peach/main.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/peach/main
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/peach.svg
        :target: https://pypi.python.org/pypi/peach
        :alt: PyPI

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/peach.svg
        :target: https://pypi.python.org/pypi/peach
        :alt: Supported Python Versions

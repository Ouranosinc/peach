============
Installation
============

..
    Stable release
    --------------
    To install portail-ing, run this command in your terminal:

    .. code-block:: console

        $ python -m pip install portail_ing

    This is the preferred method to install portail-ing, as it will always install the most recent stable release.

    If you don't have `pip`_ installed, this `Python installation guide`_ can guide
    you through the process.

    .. _pip: https://pip.pypa.io
    .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for portail-ing can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:Ouranosinc/portail-ing.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/Ouranosinc/portail-ing/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install .

However, some dependencies might be harder to install using pip. It is recommended to use mamba to create a conda environment and install the dependencies.

.. code-block:: console

    $ mamba env create -f environment.yml
    $ conda activate risk_eng
    $ pip install -e .

.. _Github repo: https://github.com/Ouranosinc/portail-ing
.. _tarball: https://github.com/Ouranosinc/portail-ing/tarball/main

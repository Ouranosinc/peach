# Installation


## Stable release

To install peach, run this command in your terminal:

```shell
python -m pip install cs-peach
```

This is the preferred method to install peach, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.


## From sources

The sources for peach can be downloaded from the [Github repo](https://github.com/Ouranosinc/peach).

You can either clone the public repository:

```shell
git clone git@github.com:Ouranosinc/peach.git
```

Or download the [tarball](https://github.com/Ouranosinc/peach/tarball/main):

```shell
curl -OJL https://github.com/Ouranosinc/peach/tarball/main
```

Once you have a copy of the source, you can install it with:

```shell
python -m pip install .
```

However, some dependencies might be harder to install using pip. It is recommended to use mamba to create a conda environment and install the dependencies.

```shell
mamba env create -f environment.yml -n peach
conda activate peach
pip install -e .
```

## Pre_workflow_data

### Description

This folder contains notebooks to access and process data required for water level indicators. The notebooks were developed by Lea Braschi @leabraschi. For more information please refer to the CBCL report presented on the [project website](https://www.ouranos.ca/fr/projets-publications/outils-analyses-risques-infrastructures).

The module `idf.py` constructs extreme precipitation indicators coded by David Huard @huard from the methodology developed by Alain Mailhot's team at INRS (for more information refer to their report on the  [project website](https://www.ouranos.ca/fr/projets-publications/outils-analyses-risques-infrastructures)).

For information on the the production of datasets used to produce general indicators, please consult the FRDR repository : https://www.frdr-dfdr.ca/repo/dataset/876e9380-63fc-4eaa-987b-aa16c3770941


### Installation Requirements

In order to run the notebooks, several libraries are required. The easiest way to get these is to run the following from the top-level directory of PEACH:

```shell
$ pip install --group data_preparation
```

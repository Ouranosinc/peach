# Usage

To use the peach risk analysis in a project, start with importing the risk package. 

```python
from peach import risk
```

Modules of interest:

`bootstrap`
: Resample data using bootstraping

`priors`
: Define the priors over models and scenarios

`xmixture`
: Defines the `XMixtureDistribution` class facilitating the evaluation of mixture statistics from `xr.DataArray` of statistical distribution parameters and weights.  


Modules `idf` and `wl` define dummy `xclim` indicators designed to access pre-computed data on the storage server. Other indicators use precipitation and temperature as input variables, run computations, and return indicator results. Sub-daily extreme precipitations and extreme hourly water levels have been pre-computed, so the indicators only need to fetch the values directly. The indicators are essentially empty placeholders storing metadata used in the frontend. 
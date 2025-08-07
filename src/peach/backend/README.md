# Backend

The following considers the backend is running on `localhost:8071`.

## API
The API is automatically documented at `http://localhost:8071/openapi?f=html`.

1. Request a computation : POST at `http://localhost:8071/processes/compute-indicators/execution`
  - with headers `{'Content-Type': 'application/json', 'Prefer' : 'respond-async'}`
  - with body : `{'inputs': {'name': '<xclim indicator id>', 'params': {... parameters}, 'stations': {'<var>': '<station id>'}}}`
    - The xclim indicator id is case-independent.
    - Add the input `"no_cache: true"` to bypass the caching and retrigger the computation.
2. The response (201) is empty except for the headers, look for `location`. It contains the URL to call to get the job's status.
3. Call the location (GET at `http://localhost:3001/jobs/<job id>`) to retrieve the status.
  - The response is JSON and contains a `status` key.
  - If `status` is "accepted" it means the job is running. Call the job status again in a moment.
  - If `status` is "successful", the job is done or failed.
  - If `status` is "failed", then try again. Most likely, the worker was terminated (because it reached it's max request number) while computing.
4. Call `http://localhost:8071/jobs/<job id>/results?f=json` to get the results in json.
  - If the `id` is "links", then `value` is an absolute path for the zarr from inside the backend's container. At the time of writing this, the top folder `workspace` corresponds to folder `workspace` in the source folder, so `Path('peach/') / value[1:]` should give the proper path.

## Configuration

See config.yml for configuration options. https://docs.pygeoapi.io/en/stable/index.html

## Installation
In env with `peach`:
```bash
pip install 'pydantic<2'
pip install pygeoapi

export PYGEOAPI_CONFIG=config.yml PYGEOAPI_OPENAPI=openapi.yml
pygeoapi openapi generate $PYGEOAPI_CONFIG > $PYGEOAPI_OPENAPI
pygeoapi serve
```

## Testing

```bash
curl -d '{"inputs":{"echoInput":"42","pause":0.1}}' -H 'Content-Type: application/json' -X POST http://localhost:5000/processes/echo/execution


curl -X 'POST' 'http://aerie.ouranos.ca:8071/processes/compute-indicators/execution' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "inputs": {
    "name": "DRY_DAYS",
    "no_cache": true,
    "params": {
      "freq": "YS-JAN",
      "thresh": "0.2 mm/d"
    },
    "stations": {
      "pr": "7033650"
    }
  }
}'

```


Standard based REST API for the backend.

Constraints on design:
- Use standard for API definition to enable interoperability.
- Assume the API will eventually be called by an HTML frontend.

## DATA

- Observations from stations
- Model projections at stations (all SSPs, all GCMs, all realizations)
- Version #

## Cache

Computed indicators, i.e. Zarr datasets on disk.

## Processes

Processes have access to Zarr data storing observations and model projections at stations for variables:
- pr
- tas
- ...

### Indicator

Compute xclim indicator for a given station over the entire dataset (obs and all model simulations).

Parameters:
- `name` : indicator ID (`indicator.identifier`, key of `xc.core.indicator.registry`)
- `params` : kwargs for the indicator
- `stations` : station ID for each variable (pr, tas)
- `no_cache` : true to recompute even if cached

Returns the file name of the annual time series of indicators at the station(s).
Cache under hash of call parameters and input data version to speed up subsequent calls.

### Probabilities for the occurrence of climate hazards

A hazard is defined as exceedance of a threshold for an indicator at a station.

Parameters:
- Computed indicator hash (for data access)
- reference period (YYYY-YYYY)
- future period (YYYY-YYYY)
- statistical distribution
- threshold (value) - interface can compute value from return period

Pseudo-code
- Access indicator time series
- Estimate distribution parameters for reference period over observations
- Compute probability of exceedance during reference period
- Estimate distribution parameters for future period over model projections
- Create mixture distribution from all model projection distributions
- Estimate exceedance probability from the mixture distribution
- Return change in risk ?

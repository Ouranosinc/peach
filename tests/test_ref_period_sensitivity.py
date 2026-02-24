"""Test the sensitivity of results to changes in the reference period.

1. Pick N stations with more than 60 years of data (start with the case study station)
2. Pick all indicators
3. Define a few quantiles of interest (.5, .9, .99)
4. Define two reference periods (first 30 years, last 30 years)
4. Define two future periods (1940-1970, 1970-2100)
4. Compute the exceedance likelihood For the various possibilities, see how the reference period affects results

"""
from pathlib import Path
import yaml
import numpy as np
from peach.common.config import read_stations
from peach.frontend import parameters as p
ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "src" / "peach" / "frontend" / "config"


def choose_stations(n=10):
    # Read stations and select those with more than 50 years of data -> 891 stations (multiple variables)
    all_stations = read_stations()
    valid = all_stations["valid_months"] / 12 >= 60
    sub = all_stations.loc[valid]

    # Stations that have both tas and pr
    st = sub.groupby(["station"]).filter(lambda x: {"tas", "pr"}.issubset(set(x["variable"]))).set_index("station")
    sids = np.array(list(set(st.index)))

    i = np.random.choice(len(sids), n, replace=False)
    sids = sids[i]

    return st.loc[sids]


def get_indicators():
    with open(CONFIG / "indicators.yml") as file:
        conf = yaml.safe_load(file)
        return conf 
        #{iid: config for iid, config in conf.items() if iid in indicator_ids}


# stations = choose_stations()

def run_trudeau_case_study():
    # Choose Trudeau station
    stations = read_stations().set_index("station")
    name = "MONTREAL__TRUDEAU_IN"
    i = stations["station_name"] == name
    station = stations.loc[i]

    # Choose indicators
    inds = get_indicators()
    indicator_ids = ["HEATING_DEGREE_DAYS"]

    sids = list(set(station.index))
    sid = sids[0]

    il = p.IndicatorList(config=inds, station_id={"tas": sid,}, backend="https://pavics.ouranos.ca/portail-ing-backend/")
        
    for ind in indicator_ids:
        il.add(ind)
    il.post_all_requests()
    # Wait...
    il.monitor_jobs()
    res = il.result_links

res = {'obs': {'1db9356d-382a-43f6-8ebf-b6555ba03a2f': 'https://minio.ouranos.ca/portail-ing/workspace/HEATING-DEGREE-DAYS_e14dcd715aaf52412f302342d032168b_obs_tas7025251_xc0-53-2.zarr'},
       'sim': {'1db9356d-382a-43f6-8ebf-b6555ba03a2f': 'https://minio.ouranos.ca/portail-ing/workspace/HEATING-DEGREE-DAYS_e14dcd715aaf52412f302342d032168b_sim_tas7025251_xc0-53-2.zarr'}}

k = '1db9356d-382a-43f6-8ebf-b6555ba03a2f'

# Define the quantiles
# qs = [.5, .9, 99]
r1 = (1950, 1979)
r2 = (1980, 2009)
for ref in (r1, r2):
    a = p.Analysis(level=0.05, ref_period=ref, fut_period=(2050, 2079))
    a._load_results(links=res)
    print(a.obs[k].best_dist())
    hm = p.HazardMatrix(analysis=a)
    ht = p.HazardThreshold(obs=a.obs[k], ref=a.ref[k], fut=a.fut[k], input="X", value=4345, hm=hm)
    print(ht.to_dict())


"""
In this case, there is only a 5% difference in the fut_sf value due to the change in reference period.
"""





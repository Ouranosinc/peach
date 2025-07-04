# testeur
import argparse
import random
import time
from multiprocessing import Pool
from pathlib import Path
from threading import Event, Thread

import docker
import matplotlib.pyplot as plt
import pandas as pd
import requests as req
import xclim as xc

stations = [
    "6140954",
    "7033650",
    "2400573",
    "8103050",
    "3063165",
    "501A7AR",
    "4028038",
    "2403855",
]
url = "http://localhost:3001/processes/compute-indicators/execution"
funcs = ["DLYFRZTHW", "TG_MEAN", "TX_DAYS_ABOVE"]
names = {"YS": "annual", "QS-DEC": "seasonal", "YS-JUL": "annual-indexed"}


def make_request(payload):
    headers = {"Content-Type": "application/json"}
    notsync = payload.pop("async")
    if notsync:
        headers["Prefer"] = "respond-async"
    t0 = time.perf_counter()
    elapsed = []
    try:
        res = req.post(url, json=payload, headers=headers, timeout=60)
        elapsed.append(res.elapsed.total_seconds() * 1000)
        if notsync:
            loc = res.headers["location"]
            for i in range(120):
                time.sleep(1)
                res = req.get(loc)
                elapsed.append(res.elapsed.total_seconds() * 1000)
                resd = res.json()
                if "status" not in resd:
                    print("ERROR", resd)
                if resd["status"] == "successful":
                    break
                if resd["status"] == "failed":
                    raise ValueError(resd["message"])
            else:
                print(
                    f"ERROR : Request for {payload['inputs']['params']['base']} is taking more than 120s to complete. See {loc}"
                )
                t1 = time.perf_counter()
                return {
                    "payload": payload,
                    "requests": elapsed,
                    "compute": -1,
                    "wall": t1 - t0,
                }
            res = req.get(loc + "/results?f=json")
            elapsed.append(res.elapsed.total_seconds() * 1000)
        resd = res.json()
        print(resd)
    except Exception as err:
        print(f"ERROR: {err}")
        resd = {"time": -1}
    t1 = time.perf_counter()
    return {
        "payload": payload,
        "requests": elapsed,
        "compute": resd["time"],
        "wall": t1 - t0,
    }


def make_payload(i, notsync, israndom):
    if israndom:
        func = random.choice(funcs)
        xcind = xc.core.indicator.registry[func].get_instance()
        params = {"freq": random.choice(list(names.keys()))}
        stns = {
            varname: random.choice(stations)
            for varname, param in xcind.parameters.items()
            if param.kind == 0
        }
    else:
        case = i % 3
        func = funcs[(i // 3) % len(funcs)]
        xcind = xc.core.indicator.registry[func].get_instance()
        if case == 0:
            params = {"freq": "YS"}
        elif case == 1 or "indexer" not in xcind.parameters:
            params = {"freq": "QS-DEC"}
        else:
            params = {"freq": "YS-JUL", "month": [7, 8, 9, 10]}

        stns = {
            varname: stations[i // (len(funcs) * 3)]
            for varname, param in xcind.parameters.items()
            if param.kind == 0
        }
    return {
        "inputs": {"name": func, "params": params, "stations": stns, "no_cache": True},
        "async": notsync,
    }


def log_docker_stats(closer, logfile):
    dockerClient = docker.DockerClient()
    back = dockerClient.containers.get("portail-ing-backend-dev-1")
    with Path(logfile).with_suffix(".system.csv").open("w") as f:
        f.write("temps,mem,cpu\n")
        print("Logging started.")
        for status in back.stats(decode="utf8"):
            try:
                # Calculate the change for the cpu usage of the container in between readings
                # Taking in to account the amount of cores the CPU has
                cpuDelta = (
                    status["cpu_stats"]["cpu_usage"]["total_usage"]
                    - status["precpu_stats"]["cpu_usage"]["total_usage"]
                )
                systemDelta = (
                    status["cpu_stats"]["system_cpu_usage"]
                    - status["precpu_stats"]["system_cpu_usage"]
                )
                # print("systemDelta: "+str(systemDelta)+" cpuDelta: "+str(cpuDelta))
                cpuPercent = (
                    (cpuDelta / systemDelta)
                    * (status["cpu_stats"]["online_cpus"])
                    * 100
                )
                cpuPercent = cpuPercent
                # print("cpuPercent: "+str(cpuPercent)+"%")
                # Fetch the memory consumption for the container
                mem = status["memory_stats"]["usage"]
                mem = mem / 1000000
                f.write(f"{status['read']},{mem},{cpuPercent}\n")
            except KeyError as err:
                print(f"Can't find stat {err}.")

            if closer.is_set():
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Nombre de requêtes simultanées",
    )
    parser.add_argument(
        "-m", "--num-requests", type=int, default=10, help="Nombre de requêtes totales"
    )
    parser.add_argument("-a", "--notsync", action="store_true", help="Faire en async")
    parser.add_argument(
        "-o", "--output", type=str, default="backend_stats", help="Fichier de stats."
    )
    parser.add_argument(
        "-r", "--random", action="store_true", help="Choix des requêtes aléatoire"
    )
    args = parser.parse_args()

    closer = Event()
    plog = Thread(target=log_docker_stats, args=(closer, args.output))
    plog.start()
    time.sleep(1)

    payloads = [
        make_payload(i, args.notsync, args.random) for i in range(args.num_requests)
    ]
    stats = []
    with Pool(args.num_workers) as p:
        for i, res in enumerate(p.imap_unordered(make_request, payloads)):
            reqn = min(res["requests"])
            reqx = max(res["requests"])
            reqg = sum(res["requests"]) / len(res["requests"])
            reqstr = f"{res['requests'][0]:.0f} ms {reqn:.0f} < {reqg:.0f} < {reqx:.0f} ({len(res['requests'])})"
            func = f"{res['payload']['inputs']['name']}-{names[res['payload']['inputs']['params']['freq']]}"
            print(
                f"{i:02d} Func: {func:40s} Compute: {res['compute']: 5.1f} Wall: {res['wall']: 5.1f} Requests: {reqstr}"
            )

            stats.append(
                {
                    "idx": i,
                    "func": func,
                    "wall": res["wall"],
                    "first": res["requests"][0],
                    "longest": reqx,
                }
            )

    time.sleep(1)
    closer.set()
    plog.join()

    dfreqs = pd.DataFrame.from_records(stats).set_index("idx")
    dfreqs.to_csv(Path(args.output).with_suffix(".times.csv"))
    mean_times = dfreqs.groupby("func").wall.mean()
    print()
    print("Average request times :")
    print(mean_times.sort_values().tail())

    df = pd.read_csv(
        Path(args.output).with_suffix(".system.csv"), parse_dates=["temps"]
    ).set_index("temps")

    fig, axs = plt.subplots(2, 1, sharex=True)
    df.mem.plot(ax=axs[0])
    axs[0].set_ylabel("Mem")
    df.cpu.plot(ax=axs[1])
    axs[1].set_ylabel("Cpu")
    fig.savefig(Path(args.output).with_suffix(".plot.png"))

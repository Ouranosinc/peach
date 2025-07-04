---
file_format: mystnb
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
execution-mode: cache
---

# Services de calculs

Le projet "Outils facilitant les analyses des risques aux infrastructures posés par le climat" vise à faciliter l'estimation des probabilités d'aléas climatiques en climat futur. L'objectif principal du projet est de proposer une [méthode]() permettant d'inclure les principales incertitudes climatiques, et faire en sorte que les praticiens n'aient pas à faire des choix difficiles concernant la sélection de modèles climatiques ou de scénarios de GES.

Cette collection de notebooks propose des exemples d'utilisation des outils développés dans le cadre du projet, et cible des professionnels avec des aptitudes de programmation en Python.

## Services de calculs offerts

Dans le cadre du projet, différents [services de calculs](https://pavics.ouranos.ca/portail-ing-backend/) ont été développés. Ces services sont accessibles publiquement et gratuitement via le standard [OGC API Processes](https://ogcapi.ogc.org/processes/):

``compute-indicators-obs``
: Calcule un indicateur climatique sur une série d'observations climatiques provenant d'une station météorologique donnée. Retourne un lien vers les résultats en format zarr.

``compute-indicators-sim``
: Calcule des indicateurs climatiques sur une série de simulations climatiques (1950–2100) dont les biais par rapport à une station donnée a été corrigé. Retourne un lien vers les résultats en format zarr.

``compute-hazard-thresholds``
: Calcule la probabilité de dépassement de seuils climatiques pour différents indicateurs à une station donnée.

## Exemple de calcul d'un indicateur

Pour clarifier le fonctionnement de ces services, le premier exemple lance le calcul des jours de chauffage observé à la station McTavish au centre-ville de Montréal. Les paramètres d'entrée pour le calcul sont les suivants:

:name (str): Identifiant de l'indicateur xclim, voir la liste des indicateurs supportés [ici](https://xclim.readthedocs.io/en/stable/indicators.html), par exemple, `"heating_degree_days"`.
:params (dict): Paramètres de l'indicateur, par exemple, `{"thresh": "10 degC"}`.
:stations (dict): Numéro de station pour toutes les variables nécessaires au calcul de l'indicateur, par exemple, `{"tas": "7024745"}`.

Pour lancer les calculs, il faut simplement passer une commande au serveur avec les paramètres des calculs.

```{code-cell} python3
import requests
import json

process = "compute-indicators-obs"
headers = {"Content-Type": "application/json", "Prefer": "respond-sync"}
url = f"https://notos.ouranos.ca/portail-ing-backend/processes/{process}/execution"
data = {"inputs":{
    "name": "HEATING_DEGREE_DAYS",
    "params": {"thresh": "10 degC"},
    "stations": {"tas": "7024745"}
}}
resp = requests.post(url, json=data, headers=headers, timeout=60)
print(resp.headers["location"])
```
On peut ensuite consulter le statut de la tâche en cours en suivant le lien ci-dessus. On accède au résultat de la tâche avec la commande suivante:

```{code-cell} python3
results = requests.get(resp.headers['location'] + "/results?f=json").json()
print(json.dumps(results, indent=2))
```

Le lien qui est retourné pointe vers un fichier [zarr](https://zarr.dev/) hébergé sur une instance de [Minio](https://min.io/), un service web compatible avec le standard S3. Pour ouvrir et lire le fichier, on utilise les librairies `s3fs` pour accéder au systèmes de fichiers, et `xarray` pour lire le format Zarr:

```{code-cell} python3
import s3fs
import xarray as xr

dataurl = requests.utils.parse_url(results['value'])
fs = s3fs.S3FileSystem(
    endpoint_url=f"{dataurl.scheme}://{dataurl.hostname}",
    anon=True,
)
store = fs.get_mapper(dataurl.path)
ds = xr.open_zarr(store, decode_timedelta=False)
ds
```

On peut aussi utiliser l'utilitaire `mc`, fourni par Minio pour télécharger directement le dossier zarr sur la machine locale ([](../storage.md)). On l'ouvrira par la suite avec `xr.open_zarr("dossier.zarr")` dans un interpréteur python.


## Exemple de calcul des probabilités de dépassement

Le portail offre aussi un service de calcul "tout-en-un" qui réplique les résultats offert à l'onglet « Seuils climatiques » de l'interface web. L'idée est la même qu'à la section précédente, mais le résultat est un dictionnaire des probabilités de dépassement. Les données d'entrées nécessaires sont :

:indicators (List[Dict]): Une liste d'indicateurs et de leur paramètres, c'est à dire une liste de dictionnaires avec les clefs `name` et `params` tels que définies dans la section précédente.
:stations (Dict): Le même argument `stations` que précédemment.
:hazards (List[List[Dict]]): Pour chaque indicateur, une liste d'aléas, chacuns définis soit par une valeur de dépassement `X`, soit par une période de retour `T`. Une `description` optionnelle peut aussi être donnée.
:analysis (Dict): Certains paramètres modifiant l'analyse statistique. Tous sont optionnels.

Les options disponibles pour `analysis` sont:

:ref_period (tuple): Le début et la fin de la période de référence (en années). Défaut : `(1991, 2020)`.
:fut_period (tuple): Le début et la fin de la période futur étudiée (en années). Défaut : `(2041, 2070)`.
:dist (List[str]): Le nom des distributions statistiques à utiliser pour chaque indicateurs. La valeur par défaut dépend de l'indicateur.


```{code-cell} python3
import requests

process = "compute-hazard-thresholds"
headers = {"Content-Type": "application/json", "Prefer": "respond-sync"}
url = f"https://pavics.ouranos.ca/portail-ing-backend/processes/{process}/execution"
data = {"inputs": {
  "indicators": [
    {
      "name": "HEATING_DEGREE_DAYS",
      "params": {"thresh": "10 degC"},
    },
    {
      "name": "COOLING_DEGREE_DAYS",
      "params": {"thresh": "25 degC"}
    }
  ],
  "stations": {"tas": "7024745"},
  "analysis": {"fut_period": (2071, 2100)},
  "hazards": [
    [  # Pour HEATING_DEGREE_DAYS
      {"description": "Usine A", "X": 4000},
      {"description": "Usine B", "T": 20}
    ],
    [ # Pour COOLING_DEGREE_DAYS
       {"description": "Usine A", "X": 40},
       {"description": "Usine B", "T": 2}
    ]
  ]
}}
resp = requests.post(url, json=data, headers=headers, timeout=60)
print(resp.headers["location"])
```

Les résultats sont directement dans le json final:

```{code-cell} python3
results = requests.get(resp.headers['location'] + "/results?f=json").json()
print(json.dumps(results, indent=2))
```

Notez qu'il est préférable de lancer des requêtes en mode asynchrone afin d'éviter des problèmes de _timeout_ si le serveur est sollicité:
```
headers = {"Content-Type": "application/json", "Prefer": "respond-async"}
```
Si les calculs ne sont pas démarrés, le serveur retourne simplement comme résultat:
```
{'code': 'ResultNotReady',
 'type': 'ResultNotReady',
 'description': 'job accepted but not yet running'}
```


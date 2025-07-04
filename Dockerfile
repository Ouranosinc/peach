
FROM mambaorg/micromamba:1.5.9

### ENVIRONMENT and BUILD ARGS ###

# If you want multiple mamba environments, set this to 1.
ARG MULTIMAMBA=1
# One of : backend, frontend, base:
ENV ENV_NAME=base

ARG PYGEOAPI_SOURCE=https://github.com/Ouranosinc/pygeoapi/archive/refs/heads/add-job-before-accepting.zip

ARG NEW_MAMBA_USER=root
ARG NEW_MAMBA_USER_ID=0
ARG NEW_MAMBA_USER_GID=0

ARG MAMBA_ENV_FILE=environment-full.yml
ARG MAMBA_ENV_BACKEND_FILE=environment-backend-full.yml
ARG MAMBA_ENV_FRONTEND_FILE=environment-frontend-full.yml

# Set backend or frontend
ENV SERVICE=backend

# Auto-reload apps during development.
ENV RELOAD=0

# ports:
ENV PORT_INTERNAL=${PORT:-80}
ENV PORT_EXTERNAL=80

### APT (ubuntu/debian based) ###

USER root
RUN apt-get update && apt-get install -y inotify-tools procps
USER $MAMBA_USER

### MAMBA ###
# Set the user, if needed. This overrides the default `mambauser` setting.
#  This is required for volumes to have appropriate write permissions.

# we need read-write access to the volume, but the default user (mambauser) does not have it.
# we can either run the image as root (not the safest), the host user, or change the folder permissions.
# I find running as the host user the least invasive.

# Code from https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#changing-the-user-id-or-name
USER root
RUN if [ "${NEW_MAMBA_USER}" != "${MAMBA_USER}" ] && [ "${NEW_MAMBA_USER}" != "root" ]; then \
  if grep -q '^ID=alpine$' /etc/os-release; then \
  # alpine does not have usermod/groupmod
  apk add --no-cache --virtual temp-packages shadow; \
  fi && \
  usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
  --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
  groupmod "--new-name=${NEW_MAMBA_USER}" \
  "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
  if grep -q '^ID=alpine$' /etc/os-release; then \
  # remove the packages that were only needed for usermod/groupmod
  apk del temp-packages; \
  fi && \
  # Update the expected value of MAMBA_USER for the
  # _entrypoint.sh consistency check.
  echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && : ; \
  elif  [ "${NEW_MAMBA_USER}" = "root" ]; then \
  echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && : ; \
  fi;

ENV MAMBA_USER=$NEW_MAMBA_USER
USER $MAMBA_USER

# Environment Variables:
# The environment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1
ENV PORTAIL_ING_DIR /app/

WORKDIR /app

COPY --chown=${MAMBA_USER}:${MAMBA_USER} environment*.yml /app/

ARG ENV_DEFAULT=
# Install mamba dependencies
# use environment{-frontend|-backend}-full.yml
# Always clean build artifacts, to ensure a small build
RUN if [ "${MULTIMAMBA}" = "1" ]; then \
  micromamba create -y -n backend -f ${MAMBA_ENV_BACKEND_FILE} && \
  micromamba create -y -n frontend -f ${MAMBA_ENV_FRONTEND_FILE}; \
else \
  micromamba install -y -n base -f ${MAMBA_ENV_FILE}; \
fi && \
  if [ "${PYGEOAPI_SOURCE}" = "conda" ]; then micromamba install -y -n base -c conda-forge "pygeoapi>=0.17"; fi && \
  micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
### PIP ###
# PROD: use "--no-deps"
# DEV:  use "--no-deps --editable"

RUN \
  if [ "${PYGEOAPI_SOURCE}" ] && [ "${PYGEOAPI_SOURCE}" != "conda" ]; then \
    if [ "${PYGEOAPI_SOURCE}" != "pip" ]; then \
      if [ "${MULTIMAMBA}" = "1" ]; then \
        micromamba run -n backend pip install ${PYGEOAPI_SOURCE}; \
      else \
        pip install ${PYGEOAPI_SOURCE}; \
      fi; \
    else \
      if [ "${MULTIMAMBA}" = "1" ]; then \
        micromamba run -n backend pip install pygeoapi; \
      else \
        pip install pygeoapi; \
      fi; \
    fi; \
  fi
# NB: we copy src towards the end of the build, so that the
#     we do not need to re-install dependencies on each build, if they have not changed.

COPY --chown=${MAMBA_USER}:${MAMBA_USER} pyproject.toml *.rst LICENSE /app/
#COPY --chown=${MAMBA_USER}:${MAMBA_USER} src /app/src
#RUN pip install ${PIP_ARGS} .

### Panel Args: ###
ENV PANEL_FILES=/app/src/portail_ing/frontend/gui_serve.py
ENV PANEL_ARGS=""
ENV PANEL_WORKERS="${PANEL_WORKERS:-5}"
ENV PANEL_THREADS="${PANEL_THREADS:-5}"
### PyGeoAPI Args: ###
# Plugin Environment Variables:
# concatenated zarr file for station data
ENV DATASET="/data/data.zarr"
# workspace for cache, etc.
ENV WORKSPACE="/workspace"

# PyGeoAPI Environment Variables:
# Config files:
ENV PYGEOAPI_CONFIG=/app/src/portail_ing/backend/config.yml PYGEOAPI_OPENAPI=/app/open_api.yaml
# URL (for internal links)
ENV PYGEOAPI_BASE_URL="localhost:${PORT_EXTERNAL}"
ENV PYGEOAPI_HOST=0.0.0.0 PYGEOAPI_PORT=${PORT_INTERNAL}
ENV PYGEOAPI_PREFIX="api"
ENV USE_LOCAL_CACHE=1

# Gunicorn Environment Variables:
ENV WORKERS="${WORKERS:-9}"
ENV GUNICORN_ARGS="--max-requests 60 --graceful-timeout 100"

# pygeoapi.flask_app:APP
ENV GUNICORN="gunicorn  pygeoapi.starlette_app:APP -k uvicorn.workers.UvicornH11Worker -c /app/src/portail_ing/backend/gunicorn.conf.py -w ${WORKERS} -b ${PYGEOAPI_HOST}:${PYGEOAPI_PORT} ${GUNICORN_ARGS}"

COPY --chown=${MAMBA_USER}:${MAMBA_USER} run.sh /app/
CMD bash -c ./run.sh

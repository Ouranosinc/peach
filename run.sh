#!/bin/sh -x
# Command for use with docker-compose.yml.
#
# Also used when deployed to production so any changes here should also be
# tested in a PAVICS deployment.
#
# "-x" used for logging on production side.

if [ "${RELOAD}" = "1" ]; then
    pip install --no-deps -e .
else
    pip install --no-deps .
fi

if [ "${SERVICE}" = "backend" ]; then
    export BACKEND_URL="http://localhost:${PYGEOAPI_PORT}/${PYGEOAPI_PREFIX:-}"
    OTHER_SERVICE="frontend"
    if [ -z "${OBS_PATTERN}" ]; then
      echo "OBS_PATTERN not set. Setting to OBS_{var}.zarr";
      export OBS_PATTERN="OBS_{var}.zarr";
    fi;

    if [ -z "${SIM_PATTERN}" ]; then
      echo "SIM_PATTERN not set. Setting to portail_ing_{var}_CMIP6_stations_AHCCD_concat.zarr";
      export SIM_PATTERN="portail_ing_{var}_CMIP6_stations_AHCCD_concat.zarr";
    fi;

    my_function () {
        ${GUNICORN}
    }

    pygeoapi openapi generate ${PYGEOAPI_CONFIG} --format yaml --output-file ${PYGEOAPI_OPENAPI};

elif [ "${SERVICE}" = "frontend" ]; then
    OTHER_SERVICE="backend"
    if [ "${RELOAD}" = "1" ]; then
        PANEL_WORKERS=1
    fi
    my_function () {
        SETUP_SCRIPT="src/peach/frontend/warm.py"

        python -m panel serve ${PANEL_FILES} \
            --warm \
            --setup ${SETUP_SCRIPT} \
            --address 0.0.0.0 \
            --port ${PORT_INTERNAL} \
            --allow-websocket-origin=* \
            --num-procs ${PANEL_WORKERS} \
            --num-threads ${PANEL_THREADS} \
            ${PANEL_ARGS};
    }
fi

if [ "${RELOAD}" = "1" ]; then
my_function &
RUN_PID=$!
inotifywait -q -e close_write,create,move,delete -m -r /app/src | while
    read -r directory action file
    do
        case ${directory} in
          *$OTHER_SERVICE* ) echo "RELOAD WARNING: ${OTHER_SERVICE} may be restarting !" ;;
          *__pycache__*    ) echo "RELOAD WARNING: Pycache updated." ;;
          *                )
            echo "File ${file} in ${directory} has been ${action}. Killing ${SERVICE} App...";
            (
                STATUS="$(pkill -P $RUN_PID; echo $?)";
                # if pkill completed, or it could not find RUN_PID... (i.e. was not terminated)
                # then run my_func
                (test $STATUS -eq 0 || test $STATUS -eq 1) && sleep 5 && my_function
            ) &
            RUN_PID=$!
            sleep 0.1
        ;;
        esac


    done
else
    my_function;
fi

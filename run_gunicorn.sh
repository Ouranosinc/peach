RUN_ONCE=0;

my_function () {
    ${GUNICORN}
}

pygeoapi openapi generate ${PYGEOAPI_CONFIG} --format yaml --output-file ${PYGEOAPI_OPENAPI};

my_function &
if [ "${RELOAD}" = "1" ];
then
cache=""
inotifywait -q -e close_write,create,move,delete -m -r /app/src/portail_ing/backend --exclude "(__pycache__/.*|.*\.ipynb)"  | while
    read -r directory action file
    do
        echo "File ${file} in ${directory} has been ${action}. Killing Gunicorn...";
        pkill gunicorn;
        RUN_ONCE=1;
        my_function &
    done
fi

"""
Gunicorn config file.

See: https://docs.gunicorn.org/en/stable/settings.html

Specific changes in Portail Ingénieurs:
- Callbacks are added for importing xclim and xscen on server start and worker creation
- When workers have completed a request, garbage collection is triggered.

"""

preload_app = True


def on_starting(server):
    """Called prior to the server starting."""
    pass


def when_ready(server):
    """Called when the server is ready."""
    import logging  # noqa: F401

    import xclim  # noqa: F401
    import xscen  # noqa: F401

    server.log.info("[gunicorn.when_ready] xclim and xscen imported")
    logging.getLogger("botocore.httpchecksum").setLevel(logging.ERROR)
    logging.getLogger("pygeoapi.l10n").setLevel(logging.ERROR)
    pass


def pre_fork(server, worker):
    """Called prior to forking a worker."""
    pass


def post_fork(server, worker):
    """Called just after forking a worker."""
    pass


# NOTE: The following functions do not run when Gunicorn runs with Uvicorn workers.


def post_worker_init(worker):
    """Called after a worker has initialized the application."""
    import xclim  # noqa: F401
    import xscen  # noqa: F401

    worker.log.info("[gunicorn.post_worker_init] xclim and xscen imported")
    pass


def post_request(worker, req, environ, resp):
    """Called after a request has been processed by a worker."""
    import gc

    gc.collect()
    worker.log.info("[gunicorn.post_request] garbage collected (request). Did it help?")
    pass


def worker_exit(server, worker):
    """Called when a worker exits."""
    import gc

    gc.collect()
    server.log.info("[gunicorn.worker_exit] garbage collected (exit). Did it help?")
    pass


def worker_int(worker):
    """Called when a worker receives an INT or QUIT signal."""
    worker.log.info("worker received INT or QUIT signal")

    ## get traceback info
    import sys
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    worker.log.debug("\n".join(code))


def worker_abort(worker):
    """Called when a worker receives an ABRT signal."""
    worker.log.info("worker received SIGABRT signal")

# Request
# A simple class to handle requesting to the async backend of peach
import logging
import time
from enum import IntEnum
from urllib.parse import urljoin, urlparse

import requests
from param import Boolean, Integer, Parameterized, Selector, String

logger = logging.getLogger(__name__)


class JobState(IntEnum):
    unsent = -1
    submit_failed = 10
    failed = 20
    accepted = 30
    in_progress = 40
    successful = 50


class AsyncJob(Parameterized):
    """Make a single async request."""

    progress = Integer(0, bounds=(0, 100))
    state = Selector(default=JobState.unsent, objects=list(JobState))
    response_url = String("")
    active = Boolean(False)

    def __init__(self, headers: dict = None, max_retries: int = 0):
        r"""Create an empty async request to a backend.

        Send the request by calling py:meth:`post(data)`.

        Parameters
        ----------
        headers: dict, optional
            Headers for the request. Defaults are Content-Type: "application/json" and Prefer: "respond-async".
            The passed dict is merged with the defaults.
        max_retries : int
            Number of maximum automatic retries when the job fails.
            Failed submission is not retried.
        """
        super().__init__()
        self.headers = {
            "Content-Type": "application/json",
            "Prefer": "respond-async",
        } | (headers or {})
        self.max_retries = max_retries
        self.retries = 0
        self.monitor_url = None
        self.last_job = {}

    def post(self, backend, process, data):
        """Make the post request to the given process, with the given data."""
        url = urljoin(backend, f"processes/{process}/execution")
        resp = requests.post(url, json=data, headers=self.headers, timeout=60)
        try:
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Job submission failed with {url} : {e}")
            self.state = JobState.submit_failed
            self.active = False
            raise
        # else
        url_monit = resp.headers["location"]
        if urlparse(url_monit).netloc == "":
            # if location is relative, make it absolute:
            url_monit = urljoin(backend, url_monit)
        self.monitor_url = url_monit
        self.last_job = {"backend": backend, "process": process, "data": data}
        logger.info(f"Job {self.jobid} accepted from computation.")
        self.state = JobState.accepted
        self.active = True

    @property
    def jobid(self):
        if self.monitor_url is None:
            raise ValueError("Job not started yet.")
        return urlparse(self.monitor_url).path.split("/")[-1]

    def monitor(self):
        """Check the request's progress.

        Returns the state of the Job.
        """
        if self.state in [
            JobState.failed,
            JobState.successful,
            JobState.submit_failed,
            JobState.unsent,
        ]:
            return self.state
        # offset=0 is only to squash a warning of Pygeoapi.
        r = requests.get(self.monitor_url + "?f=json&offset=0").json()
        if r["progress"] != self.progress:
            self.progress = r["progress"]

        if r["status"] == "successful":
            self.response_url = self.monitor_url + "/results?f=json"
            logger.info(f"Job {self.jobid} completed. Results at {self.response_url}")
            self.state = JobState.successful
            self.active = False

        # Job has failed, maybe try again
        elif r["status"] == "failed":
            if self.retries < self.max_retries:
                logger.warning(f"Job {self.jobid} failed. Retrying. {r}")
                self.retries += 1
                self.state = JobState.unsent
                self.post(**self.last_job)
            else:
                logger.warning(
                    f"Job {self.jobid} failed. Max retries reached, stopping. {r}"
                )
                self.state = JobState.failed
                self.active = False

        elif r["status"] == "accepted" and self.state == JobState.accepted:
            logger.debug(f"Job {self.jobid} has not started yet.")
        else:
            logger.warning(f"Job {self.jobid} is in an unknown state. Response: {r}")

        return self.state

    def wait(self, interval: float = 5, timeout: float = 0):
        """Wait for a job to complete or fail.

        Parameters
        ----------
        interval : float
            Time interval between each monitoring calls (s).
        timeout : float
            Maximum time spent waiting before raising an error.
            0 means no timeout.

        """
        t0 = time.time()
        while self.state not in [JobState.successful, JobState.failed]:
            t1 = time.time()
            if timeout > 0 and (t1 - t0) > timeout:
                raise ValueError(f"Job {self.jobid} is taking too long.")
            self.monitor()
            time.sleep(interval)
        if self.state is JobState.failed:
            raise ValueError(f"Job {self.jobid} failed.")

    @property
    def result(self):
        """The result of the computation (the value of the result JSON)."""
        if self.state is not JobState.successful:
            raise AttributeError(f"Job has no result yet. State is {self.state}.")

        return requests.get(self.response_url).json()["value"]


def check_backend(backend):
    """Check if the backend is available.

    Input is the backend's hostname.

    Returns True is it is, an error message otherwise.
    """
    url = urljoin(backend, "processes")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        msg = f"Backend server ({backend}) not available: {e}"
        logger.error(msg)
        return msg
    return True

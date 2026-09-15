"""One round at a time, in a thread, asked after by id."""

import threading
import uuid

from pydantic import BaseModel

from .checks import check_servable
from .wire import (
    PredictionResponse,
    RoundRequest,
    RoundResponse,
    ServiceError,
)

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


class Job(BaseModel):
    """A round this host has been asked to run, and how far it has got."""

    id: str
    state: str = QUEUED
    #: What the host is doing, in words (``docs/adr/0031``).
    stage: str = "queued"
    done: int = 0
    total: int = 0
    result: RoundResponse | PredictionResponse | None = None
    #: The reason it failed, which is the only thing a caller can act on.
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.state in (DONE, FAILED)


class Jobs:
    """The rounds this host is running, and has run.

    In memory. See ``docs/adr/0007``.
    """

    def __init__(self, runner):
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, request, runner=None) -> Job:
        if isinstance(request, RoundRequest):
            check_servable(request.model)
        with self._lock:
            busy = next((j for j in self._jobs.values() if not j.finished), None)
            if busy is not None:
                raise BusyError(
                    f"Already running job {busy.id} ({busy.stage}). One round at a "
                    f"time: a second on the same GPU does not run slower, it runs "
                    f"out of memory and takes the first one with it."
                )
            job = Job(id=uuid.uuid4().hex[:12])
            self._jobs[job.id] = job

        work = runner or self._runner
        threading.Thread(target=self._work, args=(job, request, work), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _work(self, job: Job, request, runner) -> None:
        def report(stage: str, done: int = 0, total: int = 0) -> None:
            job.stage = stage
            job.done, job.total = done, total

        try:
            job.state = RUNNING
            job.stage = "starting"
            result = runner(request, report)
            # Set before done. docs/adr/0007
            job.result = result
            job.stage = "finished"
            job.state = DONE
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            job.stage = "failed"
            job.state = FAILED


class BusyError(ServiceError):
    """A second round asked for while one is running."""

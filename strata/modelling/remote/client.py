"""The modelling service's client: asking another host to run the round.

What is sent is built from the host's own request models, the protocol is
checked before anything is sent, a round is submitted and polled, and a
failed question about a round is not a failed round. Standard library.
See ``docs/adr/0007``.
"""

import json
import time
import urllib.error
import urllib.request

from .wire import PROTOCOL, PROTOCOL_HEADER, PredictionRequest, RoundRequest

#: The timeout for a call that names none. Every call names one, since a
#: round is submitted and polled rather than held open (``docs/adr/0007``).
DEFAULT_TIMEOUT = 4 * 3600


class RemoteError(Exception):
    """Something went wrong talking to the modelling host."""


class Refused(RemoteError):
    """The host answered, and its answer was no. Final; not retried.

    See ``docs/adr/0007``.
    """


class Unreachable(RemoteError):
    """The question did not arrive. Says nothing about the round."""


class Trainer:
    """A modelling host, reached over HTTP."""

    def __init__(self, url: str, token: str, timeout: int = DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._spoken = False
        self._health: dict = {}

    def served_catalog(self) -> dict:
        """Which catalog that host trains from: ``{name, id}``, as it says itself.

        From the same answer the protocol check reads, so asking costs
        nothing once anything else has been asked.
        """
        self._handshake()
        return self._health.get("catalog") or {}

    def _handshake(self) -> None:
        """Refuse a host on another protocol, before asking it anything.

        Once per client. See ``docs/adr/0007``.
        """
        if self._spoken:
            return
        request = urllib.request.Request(f"{self.url}/healthz", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                health = json.loads(response.read())
            spoken = health.get("protocol")
        except urllib.error.HTTPError as e:
            raise Refused(f"{self.url}/healthz refused it: {_detail(e)}") from None
        except urllib.error.URLError as e:
            raise Unreachable(f"Could not reach {self.url}: {e.reason}") from None
        if spoken != PROTOCOL:
            which = spoken if spoken is not None else "none — it predates protocol versions"
            raise Refused(
                f"{self.url} speaks protocol {which}, and this machine speaks "
                f"{PROTOCOL}. Nothing was sent. Upgrade strata-modelling there or "
                f"strata-labeller here, so the two match."
            )
        self._health = health
        self._spoken = True

    def _call(self, path: str, payload: dict | None = None, timeout: int | None = None) -> dict:
        self._handshake()
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                PROTOCOL_HEADER: str(PROTOCOL),
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            # The service puts its reason in the body. docs/adr/0007
            detail = _detail(e)
            raise Refused(f"{self.url}{path} refused it: {detail}") from None
        except urllib.error.URLError as e:
            raise Unreachable(f"Could not reach {self.url}: {e.reason}") from None

    def models(self) -> dict[str, str]:
        """What that host can serve, which is not what this one can."""
        return self._call("/models", timeout=30).get("models", {})

    def submit(self, request: RoundRequest) -> dict:
        """Ask for a round. Returns the job; the round runs after this returns."""
        return self._call("/round", request.model_dump(mode="json"), timeout=60)

    def predict(self, request: PredictionRequest) -> dict:
        """Ask for scores. Returns the job; the work runs after this returns.

        Checksums rather than paths. See ``docs/adr/0007``.
        """
        return self._call("/predict", request.model_dump(mode="json"), timeout=120)

    def run(self, run_id: int) -> dict | None:
        """One run there, or None. Its numbering, not this machine's."""
        try:
            return self._call(f"/runs/{run_id}", timeout=30)
        except Refused as e:
            if "No run" in str(e):
                return None
            raise

    def latest_run(self, dataset: str, catalog_id: str | None = None) -> dict | None:
        """The newest run there, or None. Its numbering, not this machine's.

        Scoped to a catalog when one is given. See ``docs/adr/0008``.
        """
        from urllib.parse import quote

        query = f"dataset={quote(dataset)}"
        if catalog_id is not None:
            query += f"&catalog={quote(catalog_id)}"
        try:
            return self._call(f"/runs/latest?{query}", timeout=30)
        except Refused as e:
            if "No runs over" in str(e):
                return None
            raise

    def job(self, job_id: str) -> dict:
        """How a round is getting on."""
        return self._call(f"/jobs/{job_id}", timeout=30)

    def follow(self, job_id: str, on_state=None, interval: float = 3.0, sleep=None) -> dict:
        """Poll until the round finishes, tolerating a network that does not.

        A question that fails to arrive is retried; what ends this loop is
        the host answering. See ``docs/adr/0007``.
        """
        sleep = sleep or time.sleep
        unreachable = 0
        while True:
            try:
                job = self.job(job_id)
                unreachable = 0
            except Unreachable:
                unreachable += 1
                if on_state is not None:
                    on_state({"state": "unreachable", "attempts": unreachable})
                sleep(interval)
                continue

            if on_state is not None:
                on_state(job)
            if job.get("state") in ("done", "failed"):
                return job
            sleep(interval)


def _detail(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read()).get("detail", str(error))
    except Exception:
        return str(error)

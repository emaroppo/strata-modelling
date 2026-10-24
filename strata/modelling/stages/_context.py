"""What every modelling stage starts from: the context, the host, the identities, the manifest.

``train`` takes a materialised directory and returns the run it recorded,
here or on the modelling host — the request says which by whether the
context names a host, and the record is the same shape either way.
``evaluate`` scores one side of a directory with a recorded run, by one
implementation (``docs/adr/0035``).

Requests and records are plain models. Nothing here names a catalog: a
directory is self-contained, and the remote branch sends a dataset's
identity for the host to resolve against its own (``docs/adr/0008``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from strata.contracts import MANIFEST_NAME, Manifest

from ..remote.client import Trainer
from ..store.runs import RunStore

DATASET_DIR = "dataset_dir"


RUN = "run"


METRICS = "metrics"


#: Where a stage did its work: on this machine, or on the modelling host.
Where = Literal["local", "remote"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Keyed(Strict):
    """A stage's request, dumped as it was before its later fields existed.

    An experiment keys a stage on its request's dump, defaults and all
    (``docs/adr/0037``), so a field added with a default would move every
    key a ledger already holds and orphan the records under them. A field
    named in ``later`` is left out of the dump while it holds its default:
    a request that does not use it dumps, and is keyed, as it was before
    the field existed. One that does use it is another request, and its
    key says so.
    """

    #: Fields added after the stage's keys were in ledgers.
    later: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _as_before(self, handler: SerializerFunctionWrapHandler) -> Any:
        dumped = handler(self)
        fields = type(self).model_fields
        for name in self.later:
            if getattr(self, name) == fields[name].get_default(call_default_factory=True):
                dumped.pop(name, None)
        return dumped


class StageError(Exception):
    """A request a stage cannot honour as it stands."""


@dataclass(frozen=True)
class Host:
    """A modelling host, as the settings name it."""

    url: str
    token: str


@dataclass
class Context:
    """The handles: this host's run store, and the other host if there is one."""

    #: None for a round that runs on the host, which keeps its own.
    store: RunStore | None = None
    host: Host | None = None
    #: What a model reports as it trains, forwarded; see :data:`EpochReport`.
    on_epoch: object = None
    #: A remote job's state as it is polled, for whoever is watching.
    on_state: object = None
    #: How to build the client, so a test can hand in a fake host.
    client: Callable[[str, str], Trainer] = Trainer
    #: Where what a media model generates to be scored is kept, one file per
    #: checkpoint named by its content, so a record names it rather than
    #: holding it. None keeps the outputs in the record. See
    #: ``docs/adr/0054``.
    outputs: Path | None = None


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


class DatasetIdentity(Strict):
    """What a dataset id means where the round was prepared, for a host to check."""

    dataset_id: int
    name: str
    version: int
    annotation_digest: str | None
    catalog_id: str


def _manifest(directory: Path) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise StageError(f"{directory} is not a materialised dataset: no {MANIFEST_NAME}.")
    return Manifest.model_validate_json(path.read_text())

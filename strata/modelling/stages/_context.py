"""What every modelling stage starts from: the context, the host, the identities, the manifest.

Nothing here names a catalog; the remote branch sends a dataset's identity instead. See
``docs/adr/0008``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from strata.contracts import MANIFEST_NAME, Manifest

from ..model import EpochReport
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
    """A stage's request, dumped as it was before its ``later`` fields existed.

    A field named in ``later`` is left out of the dump while it holds its
    default, so ledger keys made before it survive. See ``docs/adr/0037``.
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
    on_epoch: EpochReport | None = None
    #: A remote job's state as it is polled, for whoever is watching.
    on_state: Callable[[dict], None] | None = None
    #: How to build the client, so a test can hand in a fake host.
    client: Callable[[str, str], Trainer] = Trainer
    #: Where a media model's scored outputs are kept, one content-named file per checkpoint; None
    #: keeps them in the record (``docs/adr/0054``).
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

# strata-modelling

Training and prediction over a materialised dataset, and the record of
what was trained: model plugins, runs, metrics, checkpoints and a
prediction cache. The training core takes a directory and a manifest and
nothing else, which is what lets it be tested against a fixture directory
and run on a machine that has never heard of a catalog.

```bash
uv add strata-modelling                  # the core, no framework
uv add "strata-modelling[image]"         # torch, torchvision, timm: the image baselines
uv add "strata-modelling[text]"          # torch, transformers: the text baselines
uv add "strata-modelling[service]"       # the training service; pulls in strata-catalog
```

Depends on `strata-labels` and `strata-common[migrations]`. May import
`strata-catalog` only from its service layer, and never `strata-labeller`
or Label Studio. Asking for a baseline whose extra is not installed fails
with a message naming the extra.

## The model contract

A model implements four methods and knows nothing about where its data
came from. It is handed file paths and values, never a catalog, a database
or Label Studio.

```python
class MyModel(Model):
    task = "classification"

    def finetune(self, train, classes, val=None, on_epoch=None): ...
    def predict(self, paths, on_batch=None, *, features=None): ...
    def save(self, path): ...
    def load(self, path): ...
```

An `Example` carries a path, a target, and whatever the project declared
as a feature: something already known that the model may be told. A model
declares what it cannot predict without through `requires_features`, the
classes it emits through `requires_classes`, and a label-set shape it
cannot represent through `requires_schema`. All three are checked before
the round, so a mismatch is a refusal rather than a number reported for a
projection of the data. The first two are read from the built model, so
a requirement that depends on a parameter is set in the constructor and
a fixed one is a class attribute. `on_epoch` and `on_batch` are optional to call
and mandatory to accept, because a caller on another machine cannot
otherwise tell minute one from minute nine.

Models are found two ways. A short name resolves through the
`strata.models` entry point group; anything containing `:` is a direct
reference to a class in a file, which keeps the quick-experiment path and
is refused over HTTP.

### The baselines that ship

| `ref` | task | media | notes |
|---|---|---|---|
| `multilabel` | classification | image | several classes at once |
| `multiclass` | classification | image | mutually exclusive |
| `presence` | classification | image | carries an implicit negative class |
| `text` | classification | text | several classes at once, sigmoid |
| `text-multiclass` | classification | text | mutually exclusive, softmax |
| `text-span` | span | text | BIO tagging, with optional windowing |

The pairs differ by a handful of hooks and share the rest. Nothing ships
for boxes. `ModelContract` in `strata.modelling.plugins.conformance` is the suite
a plugin runs against itself.

## The run store

Runs, their metrics and their predictions are rows in a SQLite store, one
per project and one per modelling host. A run names its parent, the
dataset version and model version behind it, and the class list as
trained; its id is a timestamp and a host token, unique without
coordination. The prediction cache beside it is keyed on the run, the
sample's checksum and a digest of its features, and nothing in it is ever
invalidated.

| `strata-runs` | |
|---|---|
| `merge --from DIR --into DIR` | fold one run store into another, oldest first; reports before `--apply` |

```bash
STRATA_RUNS_ROOT=<project>/runs strata-modelling-migrate upgrade head
```

## The service

`strata-modelling` is the same handler reached over a wire. It accepts a
dataset's identity rather than a directory, materialises it from its own
catalog, chooses the parent from the runs it holds, trains and records. A
round is submitted and polled, one at a time, refused rather than queued.
Both sides state a protocol, checked on `/healthz` before anything is
sent. `strata.modelling.remote.clientTrainer` is the standard-library client.

The GPU host's user unit and environment example are under `deploy/gpu/`.
The token comes from `STRATA_MODELLING_TOKEN`; where the catalog is comes
from the `config.toml` that `STRATA_CONFIG` names.

## Stages

Two stage functions in `strata.modelling.stages`. `train` trains from a
materialised directory here, or on the host from the dataset's identity,
and returns the same record either way. `evaluate` scores one side of a
directory with a recorded run through one implementation: exact match,
micro precision, recall and F1, and a per-class table. Classification only
so far.

## Decisions

Recorded in the umbrella repository's `docs/adr/`: the manifest as the
contract (0004), a run as a chain in a database (0005), a prediction as a
function of three inputs (0006), one handler and two transports (0007),
features as a role (0011), and a model refusing before a round (0014).

## Tests

```bash
uv run pytest packages/modelling
```

Most of the suite runs on the base install with no framework.

"""A run with every field a store insists on already filled in.

Three suites recorded runs and each had grown its own copy of this. One
here, with the defaults the assertions rely on: dataset ``demo``, one
class, a val_accuracy of 0.5 when nothing else is said.
"""

from strata.modelling.requests import Run
from strata.modelling.runs import RunStore


def a_run(**overrides) -> Run:
    base = dict(
        id="",
        dataset="demo",
        dataset_version=1,
        label_set="demo",
        model="toy",
        model_version="1",
        classes=["cat"],
    )
    return Run(**{**base, **overrides})


def recorded(store: RunStore, metrics: dict | None = None, **overrides) -> Run:
    """``a_run`` written to the store, with its metrics."""
    return store.record(a_run(**overrides), {"val_accuracy": 0.5} if metrics is None else metrics)

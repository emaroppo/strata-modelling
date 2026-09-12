"""Which catalog a run trained against.

A dataset id is an integer, and a dataset name is a string; both mean
something only inside one catalog. A host serving two of them can answer
"the latest run over demo" two ways, and can materialise dataset 7 from
whichever one it happens to be pointed at. Neither failure raises anything
— the round succeeds, over the wrong data — which is why this is recorded
rather than assumed.
"""

import pytest
from run_factory import recorded

from strata.modelling import RunStore
from strata.modelling.remote.service import CatalogMismatch, check_catalog

A = "20260101T000000-aaaaaaaa"
B = "20260202T000000-bbbbbbbb"


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------


def test_a_run_remembers_its_catalog(tmp_path):
    store = RunStore.local(tmp_path / "runs")
    run = recorded(store, catalog_id=A)
    assert store.get(run.id).catalog_id == A


def test_the_latest_run_is_scoped_to_a_catalog(tmp_path):
    store = RunStore.local(tmp_path / "runs")
    mine = recorded(store, catalog_id=A)
    recorded(store, catalog_id=B)

    # Newest overall is the B run, and warm-starting from it would continue
    # a model trained on entirely different samples
    assert store.latest("demo", A).id == mine.id
    assert store.latest("demo").catalog_id == B


def test_a_run_from_before_identities_still_counts(tmp_path):
    store = RunStore.local(tmp_path / "runs")
    old = recorded(store, catalog_id=None)

    # Null is unknown, not foreign. Excluding it would cold start every
    # project whose whole history predates this column.
    assert store.latest("demo", A).id == old.id


def test_a_known_run_wins_over_an_unknown_one(tmp_path):
    store = RunStore.local(tmp_path / "runs")
    recorded(store, catalog_id=None)
    newer = recorded(store, catalog_id=A)
    assert store.latest("demo", A).id == newer.id


# ----------------------------------------------------------------------
# The check
# ----------------------------------------------------------------------


def test_a_mismatch_is_refused():
    with pytest.raises(CatalogMismatch, match="wrong data"):
        check_catalog(A, B)


def test_the_message_names_both():
    with pytest.raises(CatalogMismatch) as caught:
        check_catalog(A, B)
    # Which one is which is the whole content of the error: a caller has to
    # know whether to repoint itself or the host
    assert A in str(caught.value) and B in str(caught.value)


def test_agreement_passes():
    check_catalog(A, A)


def test_a_round_naming_no_catalog_is_refused():
    """Tolerated once, for clients and catalogs older than identities.

    Every catalog has one now — minted the first time it is opened — and a
    client too old to send one is refused by the protocol check before a
    round gets this far. Tolerating silence would only let through a round
    nobody can say is on the right data.
    """
    with pytest.raises(CatalogMismatch):
        check_catalog(None, B)

"""What a round is checked against before it is accepted: catalog, dataset, features, model."""

from ..plugins.registry import available
from .wire import RoundRequest, ServiceError, SplitSides


class CatalogMismatch(ServiceError):
    """A round prepared against one catalog, submitted to a host on another."""


class DatasetMismatch(ServiceError):
    """A dataset id that names other data here than where the round was prepared."""


class SplitMismatch(ServiceError):
    """A split sent for a version that lists other samples here, or in another order."""


def check_catalog(requested: str, serving: str) -> None:
    """Refuse a round prepared against a different catalog.

    ``dataset_id`` is an integer, and integers are only meaningful within
    one catalog: submitted to a host serving another, the same id names
    different samples and the round trains on the wrong data without
    failing.
    """
    if requested == serving:
        return
    raise CatalogMismatch(
        f"This host serves catalog {serving}, and the round was prepared "
        f"against {requested}. A dataset id names different samples in "
        f"each, so training it here would train on the wrong data."
    )


def check_dataset(request: RoundRequest, found) -> None:
    """Refuse a round whose dataset id names something else in this catalog.

    The catalog check is not enough alone. A copy of a catalog keeps its
    identity — that is what lets its answers be merged back — and numbers
    its datasets on its own, so dataset 12 on a laptop working from a copy
    and dataset 12 here can be different data with the catalog check
    passing. The round says what it means by the id; this checks that it
    means the same here. ``found`` is what this host's catalog says the id
    is, a :class:`~strata.catalog.DatasetRef`.
    """
    if (request.dataset_name, request.dataset_version) != (found.name, found.version):
        raise DatasetMismatch(
            f"Dataset {request.dataset_id} is {found.name} v{found.version} in this "
            f"host's catalog, and the round was prepared for {request.dataset_name} "
            f"v{request.dataset_version}. A copy of a catalog numbers its datasets on "
            f"its own, so the same id can name different data in each. Freeze the "
            f"dataset in the catalog this host reads, or merge the copy back first."
        )
    if request.annotation_digest != found.annotation_digest:
        raise DatasetMismatch(
            f"{found.name} v{found.version} has different answers in this host's "
            f"catalog than where the round was prepared — a copy labelled since it "
            f"was taken. Merge it back and freeze the dataset again."
        )


def check_split(split: SplitSides, manifest) -> list[str] | None:
    """The sides to apply to the version this host materialised, or None
    when the version already has them.

    Positional, so the order is proven first: the count, and the digest of
    the checksums in manifest order. A split that fails either would land
    sides on the wrong samples without failing, which is the fault this
    check exists for.
    """
    from strata.labels import order_digest, sides_from_string, sides_string

    if len(split.sides) != len(manifest.samples):
        raise SplitMismatch(
            f"The round's split names {len(split.sides)} sample(s), and {manifest.dataset} "
            f"v{manifest.version} has {len(manifest.samples)} here."
        )
    if split.order_digest != order_digest(manifest):
        raise SplitMismatch(
            f"The round's split is over samples in an order this host's copy of "
            f"{manifest.dataset} v{manifest.version} does not have. The sides are "
            f"positional, so nothing was applied."
        )
    try:
        sides = sides_from_string(split.sides)
    except ValueError as e:
        raise SplitMismatch(str(e)) from None
    if split.sides == sides_string(manifest):
        return None
    return sides


def check_features(request: RoundRequest) -> list:
    """The feature declarations a round carries, or a refusal naming the bad one.

    Read before anything is fetched: a declaration this host cannot act on
    would otherwise fail partway through materialising, after the expensive
    part.
    """
    from strata.catalog.versions.features import FeatureError, FeatureSpec

    try:
        return [FeatureSpec.from_dict(raw) for raw in request.features]
    except FeatureError as e:
        raise ServiceError(f"The round's feature declarations: {e}") from None


def check_servable(model: str) -> None:
    """Refuse a model this host cannot honestly resolve.

    A file reference names a path on the caller's machine. Importing it here
    would either fail confusingly or, worse, find a different file with the
    same name — and a service that imports whatever path a caller names is a
    different thing from a service that serves what it has.
    """
    if ":" not in model:
        return
    raise ServiceError(
        f"{model!r} is a direct reference to code on the caller's machine, and "
        f"this host cannot resolve it. Either register the model as a "
        f"'strata.models' entry point on this host and ask for it by name "
        f"(available here: {', '.join(sorted(available())) or 'none'}), or run "
        f"the round locally, where file references still work."
    )

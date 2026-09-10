"""Turning a model name into a class.

Two paths, and the name says which. A short name resolves through the
``strata.models`` entry point group, so a request can carry ``"multilabel"``
rather than an import path — which matters once the request crosses a wire
and the backend, not the caller, decides what it can serve. Anything
containing ``:`` is a direct reference, which keeps the quick-experiment
path: drop a ``model.py`` beside your work and point at it.
"""

import hashlib
import importlib
import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

from .model import Model

#: Where an installed distribution advertises the models it provides.
ENTRY_POINT_GROUP = "strata.models"


class ModelError(Exception):
    """A model that cannot be found, loaded, or used for the job at hand."""


def available() -> dict[str, str]:
    """Registered short names, and what each resolves to.

    What a ``models`` command reports, and the only honest answer to "what
    can this backend serve" — it is read from what is installed rather than
    from a list someone maintains.
    """
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def absolute(name: str, root: Path | None = None) -> str:
    """A reference that will still resolve without ``root`` to hand.

    A run records the model it used, and is read back later — to predict, or
    to warm-start — when the directory a file reference was relative to is
    long gone. Anchoring it at the point of use is what keeps a run
    self-describing. Registry names need no such help.
    """
    if ":" not in name:
        return name
    target, class_name = name.rsplit(":", 1)
    if not target.endswith(".py"):
        return name
    path = Path(target)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return f"{path.resolve()}:{class_name}"


def resolve(name: str, root: Path | None = None) -> type[Model]:
    """The class a model name refers to.

    ``root`` anchors a ``file.py:Class`` reference, so a project-local model
    is found relative to the work rather than the current directory.
    """
    if ":" not in name:
        return _from_registry(name)
    target, class_name = name.rsplit(":", 1)
    module = (
        _module_from_file(target, root)
        if target.endswith(".py")
        else _module_from_import(target, name)
    )
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ModelError(f"No class {class_name!r} in {target}") from None


def _from_registry(name: str) -> type[Model]:
    found = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
    if not found:
        known = ", ".join(sorted(available())) or "none installed"
        raise ModelError(
            f"No model named {name!r} is registered here (available: {known}). "
            f"Names with a ':' are treated as a direct reference instead."
        )
    entry = found[0]
    try:
        return entry.load()
    except ImportError as exc:
        # The common case is a baseline asked for by name on an install that
        # never asked for its framework — which is what a request over the
        # wire does — and it deserves the same answer as a full reference
        raise ModelError(
            f"Model {name!r} is registered as {entry.value}, which will not "
            f"import: {exc}{_extra_hint(entry.module)}"
        ) from exc


def _module_from_file(target: str, root: Path | None):
    path = Path(target)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.exists():
        raise ModelError(f"Model reference points at a missing file: {path}")

    # Keyed on the whole path, not the stem: two projects each carrying a
    # model.py are two models, and naming them alike made one silently shadow
    # the other. Returned from the cache when already loaded, so resolving
    # the same file twice yields the same class — without which a model
    # cannot even be compared to itself.
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    module_name = f"strata_modelling_local_{path.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelError(f"Could not load a model from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses and pickle inside can find it
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ModelError(
            f"{path} will not import: {exc}. A model brings its own "
            f"dependencies; install what it needs."
        ) from exc
    return module


def _module_from_import(target: str, name: str):
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        raise ModelError(
            f"Model reference {name!r} names a module that will not import: "
            f"{exc}{_extra_hint(target)}"
        ) from exc


def _extra_hint(module: str) -> str:
    """Name the extra when a baseline's framework is what is missing.

    The frameworks are optional dependencies, so the common failure here is
    not a broken model but an install that never asked for one. Saying which
    extra beats a ModuleNotFoundError from inside importlib.
    """
    from .baselines import EXTRAS

    extra = EXTRAS.get(module)
    if extra is None:
        return ""
    return (
        f". That is a baseline, and it needs the '{extra}' extra — "
        f"`uv sync --extra {extra}` in a checkout, or "
        f'`uv pip install "strata-modelling[{extra}]"`'
    )

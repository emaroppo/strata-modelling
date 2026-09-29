"""Turning a model name into a class.

Two paths, and the name says which. A short name resolves through the
``strata.models`` entry point group; anything containing ``:`` is a direct
reference, ``module:Class`` or ``file.py:Class``. See ``docs/adr/0034``.
"""

import hashlib
import importlib
import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

from strata.common import plugins

from ..model import Model

#: Where an installed distribution advertises the models it provides.
ENTRY_POINT_GROUP = "strata.models"


class ModelError(Exception):
    """A model that cannot be found, loaded, or used for the job at hand."""


def available() -> dict[str, str]:
    """Registered short names, and what each resolves to.

    Read from what is installed. See ``docs/adr/0034``.
    """
    return plugins.available(entry_points(group=ENTRY_POINT_GROUP))


def absolute(name: str, root: Path | None = None) -> str:
    """A reference that will still resolve without ``root`` to hand.

    A file reference is anchored; registry names need no such help. See
    ``docs/adr/0005``.
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
    found = _from_registry(name) if ":" not in name else _from_reference(name, root)
    _refuse_the_old_name(found, name)
    return found


def _refuse_the_old_name(model_cls: type, name: str) -> None:
    """Refuse a model that still says ``task`` for its label type.

    Otherwise it inherits the default label type and is refused later as a
    mismatch, or worse, accepted as a classifier. See ``docs/adr/0041``.
    """
    own = [c for c in model_cls.__mro__ if c is not Model and issubclass(c, Model)]
    if any("task" in vars(c) for c in own) and not any("label_type" in vars(c) for c in own):
        raise ModelError(
            f"Model {name!r} declares `task`, which is now `label_type`: the "
            f"label type it learns from, classification, span or bbox. Rename "
            f"the attribute."
        )


def _from_reference(name: str, root: Path | None) -> type[Model]:
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
    entry = plugins.find(
        entry_points(group=ENTRY_POINT_GROUP),
        name,
        what="model",
        error=ModelError,
        hint=" Names with a ':' are treated as a direct reference instead.",
    )
    try:
        return entry.load()
    except ImportError as exc:
        # A baseline asked for by name on an install without its framework
        # gets the same answer as a full reference. docs/adr/0034
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

    # Keyed on the whole path, not the stem, and returned from the cache when
    # already loaded so resolving the same file twice yields the same class.
    # docs/adr/0025
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

    See ``docs/adr/0034``.
    """
    from ..baselines import extra_for

    extra = extra_for(module)
    if extra is None:
        return ""
    return (
        f". That is a baseline, and it needs the '{extra}' extra — "
        f"`uv sync --extra {extra}` in a checkout, or "
        f'`uv pip install "strata-modelling[{extra}]"`'
    )

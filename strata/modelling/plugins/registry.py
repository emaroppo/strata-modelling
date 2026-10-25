"""Turning a model name into a class.

A short name resolves through ``strata.models``; ``module:Class`` or
``file.py:Class`` is a direct reference. See ``docs/adr/0034`` and ``docs/adr/0054``.
"""

import hashlib
import importlib
import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

from strata.common import plugins

from ..baselines import extra_for
from ..model import LabelModel, MediaModel, Model

#: Where an installed distribution advertises the models it provides.
ENTRY_POINT_GROUP = "strata.models"

#: The contract's own classes, which a plugin's declarations are read past.
_CONTRACT = (Model, LabelModel, MediaModel)


class ModelError(Exception):
    """A model that cannot be found, loaded, or used for the job at hand."""


def available() -> dict[str, str]:
    """Registered short names, and what each resolves to. See ``docs/adr/0034``."""
    return plugins.available(entry_points(group=ENTRY_POINT_GROUP))


#: A reference that resolves without ``root``: :func:`strata.common.plugins.anchor`
#: (``docs/adr/0005``, ``docs/adr/0015``).
absolute = plugins.anchor


def resolve(name: str, root: Path | None = None) -> type[Model]:
    """The class a model name refers to; ``root`` anchors a ``file.py:Class`` reference."""
    found = _from_registry(name) if ":" not in name else _from_reference(name, root)
    _refuse_the_old_name(found, name)
    _refuse_a_bare_model(found, name)
    return found


def _refuse_a_bare_model(model_cls: type, name: str) -> None:
    """Refuse a class that is neither a ``LabelModel`` nor a ``MediaModel`` (``docs/adr/0054``)."""
    if isinstance(model_cls, type) and issubclass(model_cls, (LabelModel, MediaModel)):
        return
    if isinstance(model_cls, type) and issubclass(model_cls, Model):
        raise ModelError(
            f"Model {name!r} subclasses `Model`, which no longer says what it makes. "
            f"One that learns a label set and predicts is a `LabelModel`: write "
            f"`class {model_cls.__name__}(LabelModel)`. One that learns from samples "
            f"and generates new ones is a `MediaModel`."
        )
    raise ModelError(
        f"{name!r} is not a model: a model subclasses `LabelModel` or `MediaModel` "
        f"from strata.modelling."
    )


def _refuse_the_old_name(model_cls: type, name: str) -> None:
    """Refuse a model that still says ``task`` for its label type, which would otherwise read as a
    classifier.

    See ``docs/adr/0041``.
    """
    if not isinstance(model_cls, type):
        return
    own = [c for c in model_cls.__mro__ if c not in _CONTRACT and issubclass(c, Model)]
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

    # Keyed on the whole path, so resolving one file twice yields one class. docs/adr/0025
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    module_name = f"strata_modelling_local_{path.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelError(f"Could not load a model from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec, so dataclasses and pickle inside can find it.
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
    """Name the extra when a baseline's framework is what is missing. See ``docs/adr/0034``."""
    extra = extra_for(module)
    if extra is None:
        return ""
    return (
        f". That is a baseline, and it needs the '{extra}' extra — "
        f"`uv sync --extra {extra}` in a checkout, or "
        f'`uv pip install "strata-modelling[{extra}]"`'
    )

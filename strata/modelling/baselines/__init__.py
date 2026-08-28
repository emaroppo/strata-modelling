"""Baseline models shipped with auto-labeller.

A project selects one through ``[model] ref`` in ``project.toml``, or points
at its own ``model.py`` when it needs a bespoke architecture. Importing a
baseline pulls in its framework, so they are imported on demand — and the
frameworks are optional dependencies, so a project carrying its own model
never installs them.
"""

import importlib

#: Baseline class -> the module implementing it
_MODULES: dict[str, str] = {
    "MulticlassClassifier": "strata.modelling.baselines.classifier",
    "MultiLabelClassifier": "strata.modelling.baselines.classifier",
    "PresenceClassifier": "strata.modelling.baselines.classifier",
    "TextClassifier": "strata.modelling.baselines.text_classifier",
    "TextMulticlassClassifier": "strata.modelling.baselines.text_classifier",
    "TextSpanTagger": "strata.modelling.baselines.text_classifier",
}

#: Baseline module -> the install extra carrying its framework
EXTRAS: dict[str, str] = {
    "strata.modelling.baselines.classifier": "image",
    "strata.modelling.baselines.text_classifier": "text",
}

# Spelled out so static checkers can read it without executing the module
__all__ = [
    "EXTRAS",
    "MulticlassClassifier",
    "MultiLabelClassifier",
    "PresenceClassifier",
    "TextClassifier",
    "TextMulticlassClassifier",
    "TextSpanTagger",
    "extra_hint",
]


def extra_hint(module: str) -> str | None:
    """How to install the framework a baseline module needs.

    ``None`` for anything that is not a shipped baseline: a project's own
    model brings its own dependencies and we have nothing to suggest.
    """
    extra = EXTRAS.get(module)
    if extra is None:
        return None
    return (
        f"it needs the '{extra}' extra — `uv sync --extra {extra}` in a "
        f'checkout, or `uv pip install "auto-labeller[{extra}]"`'
    )


def __getattr__(name: str):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Cannot import {name}: {extra_hint(module_name)}") from exc
    return getattr(module, name)

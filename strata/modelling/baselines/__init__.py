"""Baseline models shipped with strata.modelling.

A project selects one through ``[model] ref`` in ``project.toml``, or points
at its own ``model.py`` when it needs a bespoke architecture. Each baseline
is registered under the ``strata.models`` entry point by module, so
importing one pulls in its framework only when asked for — and the
frameworks are optional dependencies, so a project carrying its own model
never installs them.
"""

#: Baseline package -> the install extra carrying its framework
EXTRAS: dict[str, str] = {
    "strata.modelling.baselines.classifier": "image",
    "strata.modelling.baselines.text": "text",
}


def extra_for(module: str) -> str | None:
    """Which extra a baseline module needs, or None for anything that is not one."""
    return next((extra for prefix, extra in EXTRAS.items() if module.startswith(prefix)), None)

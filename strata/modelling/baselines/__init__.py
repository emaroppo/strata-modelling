"""Baseline models shipped with strata.modelling, each behind an extra. See ``docs/adr/0034``."""

#: Baseline package -> the install extra carrying its framework
EXTRAS: dict[str, str] = {
    "strata.modelling.baselines.image": "image",
    # Before its parent: the first matching prefix is the one named.
    "strata.modelling.baselines.text.gliner": "gliner",
    "strata.modelling.baselines.text.sft": "sft",
    "strata.modelling.baselines.text": "text",
}


def extra_for(module: str) -> str | None:
    """Which extra a baseline module needs, or None for anything that is not one."""
    return next((extra for prefix, extra in EXTRAS.items() if module.startswith(prefix)), None)

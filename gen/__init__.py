"""Image generation providers. Pick one with the ``IMAGE_PROVIDER``
environment variable (``fal`` — the default — or ``venice``). Every provider
module exposes ``get_model``/``get_edit_model`` and
``generate_image_async``/``edit_image_async``."""

import os
from types import ModuleType

from . import fal, venice

PROVIDERS: dict[str, ModuleType] = {"fal": fal, "venice": venice}


def get_provider_name(name: str | None = None) -> str:
    """Return the provider name, defaulting to ``$IMAGE_PROVIDER`` (then
    ``fal``). Read on every call, so it reflects the env at call time
    rather than import time."""
    name = (name or os.getenv("IMAGE_PROVIDER") or "fal").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown IMAGE_PROVIDER {name!r}; expected one of: {', '.join(PROVIDERS)}"
        )
    return name


def get_provider(name: str | None = None) -> ModuleType:
    return PROVIDERS[get_provider_name(name)]

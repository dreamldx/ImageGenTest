"""LLM providers. Pick one with the ``LLM_PROVIDER`` environment variable
(``openrouter`` — the default — or ``venice``)."""

import os

from . import openrouter, venice
from .openai_compat import Provider

PROVIDERS: dict[str, Provider] = {
    provider.name: provider for provider in (openrouter.PROVIDER, venice.PROVIDER)
}


def get_provider(name: str | None = None) -> Provider:
    """Return the provider called `name`, defaulting to ``$LLM_PROVIDER``
    (then ``openrouter``). Read on every call, so it reflects the env at
    call time rather than import time."""
    name = (name or os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER {name!r}; expected one of: {', '.join(PROVIDERS)}"
        ) from None

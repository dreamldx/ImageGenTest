"""fal.ai client for text-to-image generation.

Reads the API key from the ``FAL_KEY`` environment variable (see ``.env``)
— the same name the official ``fal_client`` SDK itself looks for — and
exposes a small wrapper around ``fal_client`` for submitting text-to-image
jobs.
"""

import os
from typing import Any

import dotenv
import fal_client
import httpx

dotenv.load_dotenv()

FAL_KEY = os.getenv("FAL_KEY")

DEFAULT_MODEL = "blackforestlabs/flux-3/klein/text-to-image"

MODELS_API_URL = "https://api.fal.ai/v1/models"

# Populated by `refresh_models_cache()` / `refresh_models_cache_async()`,
# typically called once at startup.
_models_cache: list[dict] | None = None


def _models_headers() -> dict[str, str]:
    return {"Authorization": f"Key {FAL_KEY}"} if FAL_KEY else {}


def _models_params(cursor: str | None, *, limit: int, status: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        params["cursor"] = cursor
    if status is not None:
        params["status"] = status
    return params


def list_models(*, limit: int = 100, status: str | None = "active") -> list[dict]:
    """Fetch the full fal.ai model catalog (paginating through all pages).

    See https://fal.ai/docs/platform-apis/v1/models.
    """
    models: list[dict] = []
    cursor: str | None = None
    with httpx.Client(headers=_models_headers(), timeout=30.0) as client:
        while True:
            response = client.get(
                MODELS_API_URL,
                params=_models_params(cursor, limit=limit, status=status),
            )
            response.raise_for_status()
            data = response.json()
            models.extend(data.get("models", []))
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
    return models


async def list_models_async(*, limit: int = 100, status: str | None = "active") -> list[dict]:
    """Async version of :func:`list_models`."""
    models: list[dict] = []
    cursor: str | None = None
    async with httpx.AsyncClient(headers=_models_headers(), timeout=30.0) as client:
        while True:
            response = await client.get(
                MODELS_API_URL,
                params=_models_params(cursor, limit=limit, status=status),
            )
            response.raise_for_status()
            data = response.json()
            models.extend(data.get("models", []))
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
    return models


def refresh_models_cache(*, status: str | None = "active") -> list[dict]:
    """Fetch the model list from fal.ai and cache it in-memory."""
    global _models_cache
    _models_cache = list_models(status=status)
    return _models_cache


async def refresh_models_cache_async(*, status: str | None = "active") -> list[dict]:
    """Async version of :func:`refresh_models_cache`."""
    global _models_cache
    _models_cache = await list_models_async(status=status)
    return _models_cache


def get_cached_models() -> list[dict]:
    """Return the cached model list, fetching it synchronously if needed."""
    if _models_cache is None:
        return refresh_models_cache()
    return _models_cache


def get_model() -> str:
    """Return the fal.ai model currently used as the default for image
    generation (i.e. when a caller doesn't pass an explicit `model`).

    A plain `fal.DEFAULT_MODEL` attribute read already picks up runtime
    changes, but going through this function gives callers one place to
    depend on instead of reaching into the module global directly.
    """
    return DEFAULT_MODEL


def _get_client() -> fal_client.SyncClient:
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is not set in the environment (.env)")
    return fal_client.SyncClient(key=FAL_KEY)


def _on_queue_update(update: Any) -> None:
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def generate_image(
    prompt: str,
    *,
    model: str | None = None,
    image_size: str = "square_hd",
    num_images: int = 1,
    enable_prompt_expansion: bool = False,
    enable_safety_checker: bool = False,
    **extra_arguments: Any,
) -> dict:
    """Generate image(s) from a text prompt using a fal.ai model.

    `model` defaults to the *current* `DEFAULT_MODEL` (re-read on every
    call, not frozen at import time) — plain `model: str = DEFAULT_MODEL`
    would bind the default once, at function-definition time, so a later
    change to `DEFAULT_MODEL` would silently be ignored.

    `enable_prompt_expansion` and `enable_safety_checker` both default to
    `False` here, overriding the fal.ai model's own `True` defaults: our
    prompts are already expanded by the LLM before reaching here (so we
    don't want fal.ai rewriting them again on top of that), and the safety
    checker is left off by default for this use case.

    Returns the raw JSON result from fal.ai, which includes an ``images``
    list with ``url`` entries pointing to the generated images.
    """
    model = model or DEFAULT_MODEL
    client = _get_client()
    result = client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
            **extra_arguments,
        },
        with_logs=True,
        on_queue_update=_on_queue_update,
    )
    return result


async def generate_image_async(
    prompt: str,
    *,
    model: str | None = None,
    image_size: str = "square_hd",
    num_images: int = 1,
    enable_prompt_expansion: bool = False,
    enable_safety_checker: bool = False,
    **extra_arguments: Any,
) -> dict:
    """Async version of :func:`generate_image`; see its docstring re `model`,
    `enable_prompt_expansion`, and `enable_safety_checker`."""
    model = model or DEFAULT_MODEL
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is not set in the environment (.env)")
    client = fal_client.AsyncClient(key=FAL_KEY)
    result = await client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
            **extra_arguments,
        },
        with_logs=True,
        on_queue_update=_on_queue_update,
    )
    return result


if __name__ == "__main__":
    output = generate_image("a cat astronaut floating in space, digital art")
    for image in output.get("images", []):
        print(image["url"])

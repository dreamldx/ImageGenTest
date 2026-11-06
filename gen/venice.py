"""Venice.ai client for text-to-image generation and image editing.

Reads the API key from the ``VENICE_API_KEY`` environment variable (see
``.env``) and exposes async wrappers around Venice's ``/image/generate`` and
``/image/multi-edit`` endpoints, with the same call shape and result shape as
``gen/fal.py`` so ``llm/tools.py`` can use either. Select this provider with
``IMAGE_PROVIDER=venice``.

Unlike fal.ai, Venice returns image bytes rather than hosted URLs, so results
are saved under ``data/generated/`` and the returned ``url`` entries are
absolute local file paths. Those paths can be passed back in as ``edit_image``
reference images (only files inside that directory are accepted), as can any
http(s) URL.

The default models (override with ``VENICE_IMAGE_MODEL`` /
``VENICE_EDIT_MODEL``) must be aspect-ratio models, since the size preset is
sent as ``aspect_ratio``; pixel-sized models such as ``venice-sd35`` take
``width``/``height`` instead and would reject it. See
https://venice.ai/settings/api/models for the catalog (``GET /models?type=image``
and ``?type=inpaint``).
"""

import asyncio
import base64
import os
import uuid
from pathlib import Path
from typing import Any

import dotenv
import httpx

from requestlog import log_request

dotenv.load_dotenv()

VENICE_API_KEY = os.getenv("VENICE_API_KEY")

VENICE_BASE_URL = "https://api.venice.ai/api/v1"

DEFAULT_MODEL = os.getenv("VENICE_IMAGE_MODEL", "seedream-v5-pro")

DEFAULT_EDIT_MODEL = os.getenv("VENICE_EDIT_MODEL", "seedream-v5-pro-edit")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

OUTPUT_FORMAT = "png"

REQUEST_TIMEOUT = 60.0

# fal-style `image_size` presets (what the LLM tool schema offers) mapped to
# Venice aspect ratios. Venice has no 4:3 on most models, so landscape_4_3
# maps to the closest widely supported ratio.
_ASPECT_RATIOS = {
    "square_hd": "1:1",
    "square": "1:1",
    "portrait_4_3": "3:4",
    "portrait_16_9": "9:16",
    "landscape_4_3": "3:2",
    "landscape_16_9": "16:9",
}


def get_model() -> str:
    """Return the Venice model used by default for image generation."""
    return DEFAULT_MODEL


def get_edit_model() -> str:
    """Return the Venice model used by default for `edit_image`."""
    return DEFAULT_EDIT_MODEL


def _client() -> httpx.AsyncClient:
    if not VENICE_API_KEY:
        raise RuntimeError("VENICE_API_KEY is not set in the environment (.env)")
    return httpx.AsyncClient(
        base_url=VENICE_BASE_URL,
        headers={"Authorization": f"Bearer {VENICE_API_KEY}"},
        timeout=REQUEST_TIMEOUT,
    )


def _check(response: httpx.Response) -> None:
    if response.is_success:
        return
    # Venice error bodies are `{"error": "...", "details": ...}`; surface
    # them as-is so the LLM/user can see why (bad params, 402 balance, ...).
    raise RuntimeError(f"Venice {response.request.url.path} failed ({response.status_code}): {response.text[:500]}")


def _sniff_extension(data: bytes, default: str) -> str:
    # Some models ignore the requested format (e.g. flux-3-image returned
    # JPEG for format=png), so name the file after what it actually is.
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return default


def _save(data: bytes) -> str:
    extension = _sniff_extension(data, OUTPUT_FORMAT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{uuid.uuid4().hex}.{extension}"
    path.write_bytes(data)
    return str(path)


def _image_input(reference: str) -> str:
    """Turn an `edit_image` reference into what an entry of Venice's `images` field takes:
    an http(s) URL as-is, or a file we generated earlier as base64.

    Local paths are restricted to OUTPUT_DIR so an LLM-supplied path can't
    be used to read (and ship to Venice) arbitrary files.
    """
    if reference.startswith(("http://", "https://")):
        return reference
    path = Path(reference).resolve()
    if not path.is_relative_to(OUTPUT_DIR) or not path.is_file():
        raise ValueError(f"Not a known image URL or generated image path: {reference}")
    return base64.b64encode(path.read_bytes()).decode()


async def generate_image_async(
    prompt: str,
    *,
    model: str | None = None,
    image_size: str = "square_hd",
    num_images: int = 1,
    **extra_arguments: Any,
) -> dict:
    """Generate image(s) from a text prompt using a Venice model.

    `safe_mode` is off by default (Venice would otherwise blur adult
    content), matching the safety checker being off for fal.ai.

    Returns ``{"images": [{"url": <local file path>}, ...]}``, the same
    shape as ``gen/fal.py``.
    """
    body = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "aspect_ratio": _ASPECT_RATIOS.get(image_size, "1:1"),
        "format": OUTPUT_FORMAT,
        "variants": num_images,
        "safe_mode": False,
        **extra_arguments,
    }
    with log_request("image.generate", "venice", body["model"]):
        async with _client() as client:
            response = await client.post("/image/generate", json=body)
        _check(response)
    images = response.json().get("images", [])
    return {"images": [{"url": _save(base64.b64decode(image))} for image in images]}


async def edit_image_async(
    prompt: str,
    image_urls: list[str],
    *,
    model: str | None = None,
    num_images: int = 1,
    **extra_arguments: Any,
) -> dict:
    """Edit or compose from reference image(s) using a Venice image-editing
    model, via `/image/multi-edit`.

    `image_urls` entries are http(s) URLs or previously generated local
    paths. The first is the base image, the rest act as extra layers; the
    per-model cap on how many is enforced by Venice (a 400 comes back as an
    error). The output keeps the first image's aspect ratio, so `image_size`
    (accepted via `extra_arguments` for parity with fal.ai) is ignored. The
    endpoint returns one image per call, so `num_images` > 1 makes that many
    calls.

    Returns ``{"images": [{"url": <local file path>}, ...]}``.
    """
    if not image_urls:
        raise ValueError("edit_image requires at least one reference image URL")
    extra_arguments.pop("image_size", None)
    body = {
        # Unlike /image/generate, multi-edit calls the model field `modelId`.
        "modelId": model or DEFAULT_EDIT_MODEL,
        "prompt": prompt,
        "images": [_image_input(reference) for reference in image_urls],
        "output_format": OUTPUT_FORMAT,
        "safe_mode": False,
        **extra_arguments,
    }

    async def edit_once(client: httpx.AsyncClient) -> dict:
        response = await client.post("/image/multi-edit", json=body)
        _check(response)
        return {"url": _save(response.content)}

    with log_request("image.edit", "venice", body["modelId"]):
        async with _client() as client:
            images = await asyncio.gather(*(edit_once(client) for _ in range(num_images)))
    return {"images": list(images)}

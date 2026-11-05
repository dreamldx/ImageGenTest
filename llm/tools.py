"""Tool definitions that expose gen/fal to LLMs via OpenRouter's
OpenAI-style function/tool calling.
"""

from typing import Any

from gen import fal


def _build_generate_image_tool() -> dict[str, Any]:
    # Built fresh on every call (instead of a frozen module-level literal)
    # so the advertised default always reflects the *current* `fal.get_model()`
    # — which can change at runtime — rather than whatever it was at import time.
    return {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate one or more images from a text prompt using a fal.ai "
                "text-to-image model. Returns the URLs of the generated images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed text-to-image prompt describing the desired image.",
                    },
                    "model": {
                        "type": "string",
                        "description": "fal.ai model endpoint id to use.",
                        "default": fal.get_model(),
                    },
                    "image_size": {
                        "type": "string",
                        "description": "Output image aspect ratio/size preset.",
                        "enum": [
                            "square_hd",
                            "square",
                            "portrait_4_3",
                            "portrait_16_9",
                            "landscape_4_3",
                            "landscape_16_9",
                        ],
                        "default": "square_hd",
                    },
                    "num_images": {
                        "type": "integer",
                        "description": "Number of images to generate.",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 1,
                    },
                },
                "required": ["prompt"],
            },
        },
    }


def _build_edit_image_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": (
                "Edit or compose from reference image(s) using a fal.ai "
                "image-editing model, following a text instruction. Use this "
                "instead of generate_image whenever the user wants to modify, "
                "touch up, restyle, or combine elements into an *existing* "
                "image rather than create a brand-new one. You must pass the "
                "URL(s) of the image(s) to edit yourself via `image_urls` — "
                "pick them from this conversation's history (e.g. the "
                "`image_urls` a prior generate_image/edit_image call "
                "returned); usually that's just the most recently "
                "generated/edited image, unless the user points at an "
                "earlier one or supplies their own URL. Returns the URLs of "
                "the edited images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed instruction describing the desired edit.",
                    },
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "URL(s) of the reference image(s) to edit or "
                            "compose from, taken from earlier tool results in "
                            "this conversation (or a URL the user supplied)."
                        ),
                        "minItems": 1,
                    },
                    "model": {
                        "type": "string",
                        "description": "fal.ai model endpoint id to use.",
                        "default": fal.get_edit_model(),
                    },
                    "image_size": {
                        "type": "string",
                        "description": "Output image aspect ratio/size preset.",
                        "enum": [
                            "square_hd",
                            "square",
                            "portrait_4_3",
                            "portrait_16_9",
                            "landscape_4_3",
                            "landscape_16_9",
                        ],
                        "default": "square_hd",
                    },
                    "num_images": {
                        "type": "integer",
                        "description": "Number of images to generate.",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 1,
                    },
                },
                "required": ["prompt", "image_urls"],
            },
        },
    }


def get_tools() -> list[dict[str, Any]]:
    """Every tool the LLM is allowed to call, built fresh on each call."""
    return [_build_generate_image_tool(), _build_edit_image_tool()]


async def _generate_image(arguments: dict[str, Any]) -> dict[str, Any]:
    prompt = arguments["prompt"]
    extra_arguments = {key: value for key, value in arguments.items() if key != "prompt"}
    result = await fal.generate_image_async(prompt, **extra_arguments)
    images = result.get("images", [])
    return {
        "image_urls": [image["url"] for image in images],
        "count": len(images),
    }


async def _edit_image(arguments: dict[str, Any]) -> dict[str, Any]:
    prompt = arguments["prompt"]
    image_urls = arguments.get("image_urls") or []
    if not image_urls:
        return {"error": "edit_image requires at least one image URL in image_urls"}
    extra_arguments = {
        key: value for key, value in arguments.items() if key not in ("prompt", "image_urls")
    }
    result = await fal.edit_image_async(prompt, image_urls, **extra_arguments)
    images = result.get("images", [])
    return {
        "image_urls": [image["url"] for image in images],
        "count": len(images),
    }


# Maps a tool's function name to the async callable that actually runs it.
TOOL_HANDLERS = {
    "generate_image": _generate_image,
    "edit_image": _edit_image,
}


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a single tool call by name, returning a JSON-serializable result."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return await handler(arguments)
    except Exception as exc:  # noqa: BLE001 - surface the failure back to the LLM
        return {"error": str(exc)}

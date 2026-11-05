"""OpenRouter chat-completion client, used to turn a rough idea into a
well-formed text-to-image prompt via a DeepSeek Flash model.

Reads the API key from the ``OPENROUTER_API_KEY`` environment variable (see
``.env``). The model slug defaults to ``deepseek/deepseek-v4.1-flash`` but
can be overridden with the ``OPENROUTER_MODEL`` environment variable in case
that slug ever changes on OpenRouter's end.

The request/tool-loop machinery lives in :mod:`llm.openai_compat`; this
module only supplies OpenRouter's `Provider` config and thin wrappers.
"""

import os
from typing import Any, Awaitable, Callable, Optional

import dotenv

from . import openai_compat
from .openai_compat import Provider

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")

PROVIDER = Provider(
    name="openrouter",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    api_key_env="OPENROUTER_API_KEY",
    default_model=DEFAULT_MODEL,
)

SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "Given a short idea (possibly in Chinese), rewrite it into a single, "
    "detailed, vivid English prompt suitable for a diffusion model: "
    "describe the subject, composition, lighting, style, and mood. "
    "Reply with only the final prompt, no explanations, no quotes."
)


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Call OpenRouter's chat-completions endpoint, returning the raw JSON."""
    return openai_compat.chat_completion(
        PROVIDER, messages, model=model, temperature=temperature, tools=tools, **extra_arguments
    )


async def chat_completion_async(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Async version of :func:`chat_completion`."""
    return await openai_compat.chat_completion_async(
        PROVIDER, messages, model=model, temperature=temperature, tools=tools, **extra_arguments
    )


def generate_image_prompt(
    idea: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    **extra_arguments: Any,
) -> str:
    """Turn a rough idea into a detailed text-to-image prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": idea},
    ]
    data = chat_completion(messages, model=model, temperature=temperature, **extra_arguments)
    return openai_compat.extract_message(data)["content"].strip()


async def generate_image_prompt_async(
    idea: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    **extra_arguments: Any,
) -> str:
    """Async version of :func:`generate_image_prompt`."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": idea},
    ]
    data = await chat_completion_async(messages, model=model, temperature=temperature, **extra_arguments)
    return openai_compat.extract_message(data)["content"].strip()


async def run_with_tools(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_turns: int = 4,
    on_message: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Run the tool-calling loop against OpenRouter; see
    :func:`llm.openai_compat.run_with_tools`."""
    return await openai_compat.run_with_tools(
        PROVIDER,
        messages,
        tools=tools,
        tool_handlers=tool_handlers,
        model=model,
        temperature=temperature,
        max_turns=max_turns,
        on_message=on_message,
    )


if __name__ == "__main__":
    print(generate_image_prompt("一只在太空漂浮的宇航员猫"))

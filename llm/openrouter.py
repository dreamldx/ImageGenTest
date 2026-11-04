"""OpenRouter chat-completion client, used to turn a rough idea into a
well-formed text-to-image prompt via a DeepSeek Flash model.

Reads the API key from the ``OPENROUTER_API_KEY`` environment variable (see
``.env``). The model slug defaults to ``deepseek/deepseek-v4.1-flash`` but
can be overridden with the ``OPENROUTER_MODEL`` environment variable in case
that slug ever changes on OpenRouter's end.
"""

import json
import os
from typing import Any, Awaitable, Callable, Optional

import dotenv
import httpx

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")

SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "Given a short idea (possibly in Chinese), rewrite it into a single, "
    "detailed, vivid English prompt suitable for a diffusion model: "
    "describe the subject, composition, lighting, style, and mood. "
    "Reply with only the final prompt, no explanations, no quotes."
)


def _headers() -> dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment (.env)")
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _request_body(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, **extra}
    if tools:
        body["tools"] = tools
    return body


def _extract_message(data: dict[str, Any]) -> dict[str, Any]:
    return data["choices"][0]["message"]


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Call OpenRouter's chat-completions endpoint, returning the raw JSON."""
    response = httpx.post(
        CHAT_COMPLETIONS_URL,
        headers=_headers(),
        json=_request_body(messages, model=model, temperature=temperature, tools=tools, **extra_arguments),
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


async def chat_completion_async(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Async version of :func:`chat_completion`."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            CHAT_COMPLETIONS_URL,
            headers=_headers(),
            json=_request_body(messages, model=model, temperature=temperature, tools=tools, **extra_arguments),
        )
    response.raise_for_status()
    return response.json()


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
    return _extract_message(data)["content"].strip()


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
    return _extract_message(data)["content"].strip()


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
    """Run a chat-completion loop that lets the model call the given tools.

    Repeatedly sends ``messages`` to the model; whenever it responds with
    tool calls, each one is dispatched to ``tool_handlers`` by function name
    and the (JSON-serialized) result is fed back as a ``tool`` message, until
    the model returns a plain answer (or ``max_turns`` is exhausted).

    ``messages`` (the caller-supplied history) is never passed to
    ``on_message`` — only the assistant and tool messages this call
    generates are, each right when it's appended to the conversation, so a
    caller can persist chat history incrementally (and still have it be
    accurate if a later turn fails).

    Returns the final assistant message dict (with at least a ``content``
    key).
    """
    conversation = list(messages)
    message: dict[str, Any] = {}

    for _ in range(max_turns):
        data = await chat_completion_async(conversation, model=model, temperature=temperature, tools=tools)
        message = _extract_message(data)
        conversation.append(message)
        if on_message is not None:
            await on_message(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message

        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            arguments = json.loads(function.get("arguments") or "{}")

            handler = tool_handlers.get(name)
            if handler is None:
                result: Any = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = await handler(arguments)
                except Exception as exc:  # noqa: BLE001 - surface the failure back to the LLM
                    result = {"error": str(exc)}

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            }
            conversation.append(tool_message)
            if on_message is not None:
                await on_message(tool_message)

    return message


if __name__ == "__main__":
    print(generate_image_prompt("一只在太空漂浮的宇航员猫"))

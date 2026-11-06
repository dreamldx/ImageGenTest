"""Provider-agnostic client for OpenAI-compatible chat-completion APIs
(OpenRouter, Venice.ai, ...), including the tool-calling loop.

A `Provider` bundles everything that differs between them (base URL, API
key, default model, provider-specific request fields); the request/response
handling and `run_with_tools` loop are shared.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

from requestlog import log_request


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: Optional[str]
    api_key_env: str
    default_model: str
    # Merged into every request body (e.g. Venice's `venice_parameters`).
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is not set in the environment (.env)")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def request_body(
    provider: Provider,
    messages: list[dict[str, Any]],
    *,
    model: Optional[str],
    temperature: float,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model or provider.default_model,
        "messages": messages,
        "temperature": temperature,
        **provider.extra_body,
        **extra,
    }
    if tools:
        body["tools"] = tools
    return body


def extract_message(data: dict[str, Any]) -> dict[str, Any]:
    return data["choices"][0]["message"]


def chat_completion(
    provider: Provider,
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Call the provider's chat-completions endpoint, returning the raw JSON."""
    body = request_body(provider, messages, model=model, temperature=temperature, tools=tools, **extra_arguments)
    with log_request("llm", provider.name, body["model"]):
        response = httpx.post(
            provider.chat_completions_url,
            headers=provider.headers(),
            json=body,
            timeout=60.0,
        )
        response.raise_for_status()
    return response.json()


async def chat_completion_async(
    provider: Provider,
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    tools: Optional[list[dict[str, Any]]] = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """Async version of :func:`chat_completion`."""
    body = request_body(provider, messages, model=model, temperature=temperature, tools=tools, **extra_arguments)
    with log_request("llm", provider.name, body["model"]):
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                provider.chat_completions_url,
                headers=provider.headers(),
                json=body,
            )
        response.raise_for_status()
    return response.json()


async def run_with_tools(
    provider: Provider,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]],
    model: Optional[str] = None,
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
        data = await chat_completion_async(
            provider, conversation, model=model, temperature=temperature, tools=tools
        )
        message = extract_message(data)
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

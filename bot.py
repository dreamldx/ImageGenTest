"""Orchestrator: reads prompts from channel/discord, runs the LLM (with the
generate_image tool from gen/fal exposed via llm/tools) to answer them, and
replies with any generated images back through channel/discord.

Chat history is kept in sqlite (persistent/), keyed by the Discord thread's
id, and is both read (to give the LLM prior context) and written (for every
user/assistant/tool message) around each `run_with_tools` call, so the
LLM's view of a thread's history stays accurate turn to turn.

This module only defines the handlers; see main.py for the actual entry
point (hook wiring + starting the bot).
"""

import json
from datetime import datetime
from typing import Any

import discord

import persistent
from channel import discord as discord_channel
from gen import fal
from llm import openrouter, tools

AGENT_SYSTEM_PROMPT = (
    "You are a helpful Discord assistant that can generate images. "
    "When the user asks for an image:\n"
    "1. Check whether the request is clear enough to act on. If anything "
    "important is ambiguous or underspecified (subject, style, mood, "
    "composition, color, setting, etc.) and no reasonable default is "
    "obvious, ask the user a clarifying question instead of guessing — "
    "do not call the tool yet.\n"
    "2. Once the request is clear enough, expand it into a detailed, vivid "
    "prompt before calling generate_image: elaborate on the subject, "
    "composition, lighting, color palette, art style, and mood with "
    "concrete, sensible details that fit the user's intent — never invent "
    "anything that contradicts what they asked for.\n"
    "3. Write that prompt the way the image model expects: a single "
    "descriptive English sentence or comma-separated clause list, not a "
    "conversational request.\n"
    "The generated image(s) are already shown to the user as attachments, "
    "so don't repeat the raw image URL in your reply — just briefly "
    "describe what you made, in the same language the user used."
)

# Separates the per-turn dynamic prompt from the user's actual text, so the
# model can tell them apart. The dynamic prompt is rebuilt fresh on every
# call and only ever lives in the message sent to the LLM — never persisted.
DYNAMIC_PROMPT_SEPARATOR = "\n---\n"


def _build_dynamic_prompt() -> str:
    # Note: fal.get_model() is only the *fallback* generate_image uses if
    # the tool call omits `model` — the LLM can still pass a different model
    # id in its tool call, so this isn't a guarantee of what gets used.
    now = datetime.now().astimezone()
    return (
        f"Current time: {now.isoformat(timespec='seconds')}\n"
        f"fal.ai default model for generate_image: "
        f"{fal.get_model()}"
    )


async def handle_startup() -> None:
    try:
        models = await fal.refresh_models_cache_async()
    except Exception as exc:  # noqa: BLE001 - don't crash startup over the model cache
        print(f"Failed to cache fal.ai model list: {exc}")
        return
    print(f"Cached {len(models)} fal.ai models")


def _extract_image_urls(tool_message: dict[str, Any]) -> list[str]:
    try:
        payload = json.loads(tool_message.get("content") or "{}")
    except (TypeError, ValueError):
        return []
    urls = payload.get("image_urls") if isinstance(payload, dict) else None
    return urls if isinstance(urls, list) else []


async def _is_tracked_thread(message: discord.Message) -> bool:
    """Whether `message` was posted in a thread we already have history
    for (i.e. an ongoing bot conversation, not just any random thread)."""
    if not isinstance(message.channel, discord.Thread):
        return False
    thread_row = await persistent.get_thread_by_external_id_async(str(message.channel.id))
    return thread_row is not None


async def handle_prompt(message: discord.Message, prompt: str) -> None:
    # Only engage if we're explicitly @-mentioned, or this continues a
    # thread we're already tracking — otherwise stay quiet in the channel.
    is_mentioned = discord_channel.bot.user in message.mentions
    if not is_mentioned and not await _is_tracked_thread(message):
        return

    try:
        thread = await discord_channel.resolve_thread(message, prompt)
    except discord.HTTPException as exc:
        print(f"Failed to create thread: {exc}")
        return

    # One row in `threads` per Discord thread id; its `messages` are the
    # LLM conversation history for that thread. The system prompt is never
    # persisted — it's prepended fresh on every call, so editing
    # AGENT_SYSTEM_PROMPT takes effect on existing threads too.
    thread_id = await persistent.get_or_create_thread_async(str(thread.id), title=prompt[:100])
    history = await persistent.get_messages_async(thread_id)

    # Persist only the user's raw text — the dynamic prompt below is
    # per-call and must never end up in stored history.
    user_message = {"role": "user", "content": prompt}
    await persistent.add_message_async(thread_id, user_message)

    llm_user_message = {
        "role": "user",
        "content": f"{_build_dynamic_prompt()}{DYNAMIC_PROMPT_SEPARATOR}{prompt}",
    }
    conversation = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, *history, llm_user_message]

    image_urls: list[str] = []

    async def persist(new_message: dict[str, Any]) -> None:
        # Called for every assistant/tool message run_with_tools produces,
        # right as it happens — so history is saved incrementally rather
        # than only if/when the whole call succeeds.
        await persistent.add_message_async(thread_id, new_message)
        if new_message.get("role") == "tool":
            image_urls.extend(_extract_image_urls(new_message))

    async with thread.typing():
        try:
            final_message = await openrouter.run_with_tools(
                conversation,
                tools=tools.get_tools(),
                tool_handlers=tools.TOOL_HANDLERS,
                on_message=persist,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            await discord_channel.reply_with_error(thread, exc)
            return

        reply_text = final_message.get("content") or "好了。"
        await thread.send(reply_text)
        if image_urls:
            await discord_channel.reply_with_images(thread, image_urls)

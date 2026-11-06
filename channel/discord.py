"""Discord channel: reads prompts from messages and replies with images.

This module only knows about Discord I/O. It does not call fal.ai itself —
the actual image generation is wired in by ``bot.py`` through ``on_prompt``.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

import discord
from discord.ext import commands
import dotenv

dotenv.load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Set by bot.py. Called with (message, prompt) for every non-command
# message; responsible for generating and sending back the reply.
on_prompt: Optional[Callable[[discord.Message, str], Awaitable[None]]] = None

# Set by bot.py. Called once after the bot has logged in (e.g. to warm up
# caches).
on_startup: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class _StartupEvent:
    pass


@dataclass
class _PromptEvent:
    message: discord.Message
    prompt: str


_Event = Union[_StartupEvent, _PromptEvent]

# Discord I/O (receiving) just enqueues events here; a single background
# worker task drains the queue and runs the (possibly slow, fal.ai-calling)
# callbacks one at a time, so a slow generation never blocks message intake.
_event_queue: "asyncio.Queue[_Event]" = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None
_startup_queued = False


async def _handle_event(event: _Event) -> None:
    if isinstance(event, _StartupEvent):
        if on_startup is not None:
            await on_startup()
    elif isinstance(event, _PromptEvent):
        if on_prompt is not None:
            await on_prompt(event.message, event.prompt)


async def _worker() -> None:
    while True:
        event = await _event_queue.get()
        try:
            await _handle_event(event)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on bad events
            print(f"Error handling {event!r}: {exc}")
        finally:
            _event_queue.task_done()


@bot.event
async def on_ready():
    global _worker_task, _startup_queued
    print(f"Logged in as {bot.user} (id: {bot.user.id})")

    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker())

    # on_ready can fire again on reconnect; only queue startup once.
    if not _startup_queued:
        _startup_queued = True
        await _event_queue.put(_StartupEvent())


@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send("pong")


@bot.event
async def on_message(message: discord.Message):
    # Ignore ourselves and other bots to avoid feedback loops.
    if message.author.bot:
        return

    # Let messages that look like commands (e.g. "!ping") go through the
    # normal command pipeline instead of being treated as image prompts.
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Strip the bot's own mention token so it doesn't leak into the prompt
    # text (e.g. a message that's just "@Bot 画一只猫" shouldn't send the
    # literal "<@123...>" to the LLM).
    prompt = message.content
    if bot.user is not None:
        prompt = prompt.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
    prompt = prompt.strip()
    # A caption-less image upload (e.g. "edit this" via attachment alone)
    # should still go through — only bail out if there's neither text nor
    # an image attachment to act on.
    if not prompt and not extract_image_attachment_urls(message):
        return

    # A top-level channel message starts a new thread; a message posted
    # inside an existing thread continues that thread's conversation.
    if not isinstance(message.channel, (discord.TextChannel, discord.VoiceChannel, discord.Thread)):
        return

    await _event_queue.put(_PromptEvent(message, prompt))


async def create_thread(message: discord.Message, name: str) -> discord.Thread:
    """Create a reply thread off the given message."""
    return await message.create_thread(name=name[:100] or "Generated image")


async def resolve_thread(message: discord.Message, name: str) -> discord.Thread:
    """Return the thread a reply should go in: the message's own thread if
    it was posted inside one already (continuing that conversation), or a
    newly created thread off it otherwise (starting a new conversation)."""
    if isinstance(message.channel, discord.Thread):
        return message.channel
    return await create_thread(message, name)


def extract_image_attachment_urls(message: discord.Message) -> list[str]:
    """Return the URLs of any image attachments on `message`.

    Discord attachment URLs are plain public CDN links, so these can be fed
    straight to fal.ai (e.g. as `edit_image`'s `image_urls`) without any
    extra upload step.
    """
    return [
        attachment.url
        for attachment in message.attachments
        if (attachment.content_type or "").startswith("image/")
    ]


async def upload_images(thread: discord.Thread, paths: list[str]) -> list[str]:
    """Upload local image files to the thread and return their Discord CDN
    URLs, in order. The local copies are deleted once uploaded.

    For providers that return image bytes rather than hosted URLs; the CDN
    URLs are then used like any other image URL (shown to the LLM, passed
    back as `edit_image` references). The upload message itself is how the
    images get shown to the user.
    """
    files = [discord.File(path) for path in paths]
    message = await thread.send(content=f"已生成 {len(paths)} 张图片：", files=files)
    # Discord keeps the order of `files` in `message.attachments`.
    urls = [attachment.url for attachment in message.attachments]
    for path in paths:
        os.remove(path)
    return urls


async def reply_with_images(thread: discord.Thread, image_urls: list[str]) -> None:
    """Send generated image(s) back into the given thread."""
    if not image_urls:
        await thread.send("没有生成任何图片。")
        return

    embeds = []
    for url in image_urls[:10]:  # Discord allows at most 10 embeds per message.
        embed = discord.Embed()
        embed.set_image(url=url)
        embeds.append(embed)

    await thread.send(content=f"已生成 {len(image_urls)} 张图片：", embeds=embeds)


async def reply_with_error(thread: discord.Thread, error: Exception) -> None:
    await thread.send(f"图片生成失败：{error}")


def run() -> None:
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set in the environment (.env)")
    bot.run(DISCORD_BOT_TOKEN)

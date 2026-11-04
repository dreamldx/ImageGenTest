"""Chat history storage.

Threads (conversations, e.g. one per Discord thread) and messages (one row
per LLM chat message: user/assistant/tool/system) are stored in separate
tables, linked by `thread_id`. Each message is kept as a full JSON blob
(`data`) so arbitrary OpenAI/OpenRouter-style fields — `tool_calls`,
`tool_call_id`, `reasoning_details`, etc. — round-trip losslessly, while
`role`/`content` are also broken out as plain columns for easy inspection.
"""

import asyncio
import json
from typing import Any, Optional

from . import sql
from .db import LOCK, get_connection


def get_or_create_thread(external_id: str, *, title: Optional[str] = None) -> int:
    """Return the internal thread id for `external_id`, creating it if needed."""
    conn = get_connection()
    with LOCK:
        row = conn.execute(sql.select_thread_id_by_external_id(), (external_id,)).fetchone()
        if row is not None:
            return row["id"]

        cursor = conn.execute(sql.insert_thread(), (external_id, title))
        conn.commit()
        return cursor.lastrowid


def get_thread_by_external_id(external_id: str) -> Optional[dict[str, Any]]:
    conn = get_connection()
    with LOCK:
        row = conn.execute(sql.select_thread_by_external_id(), (external_id,)).fetchone()
    return dict(row) if row is not None else None


def add_message(thread_id: int, message: dict[str, Any]) -> int:
    """Append one chat message (an OpenAI/OpenRouter-style message dict)."""
    conn = get_connection()
    with LOCK:
        cursor = conn.execute(
            sql.insert_message(),
            (
                thread_id,
                message.get("role", "user"),
                message.get("content"),
                json.dumps(message, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_messages(thread_id: int) -> list[dict[str, Any]]:
    """Return every message for a thread, oldest first, as message dicts."""
    conn = get_connection()
    with LOCK:
        rows = conn.execute(sql.select_messages_by_thread_id(), (thread_id,)).fetchall()
    return [json.loads(row["data"]) for row in rows]


# --- Async wrappers, for use from asyncio event-loop code (e.g. bot.py) ---
# SQLite I/O is fast but still blocking; these run it in a worker thread so
# an async caller never stalls the event loop.


async def get_or_create_thread_async(external_id: str, *, title: Optional[str] = None) -> int:
    return await asyncio.to_thread(get_or_create_thread, external_id, title=title)


async def get_thread_by_external_id_async(external_id: str) -> Optional[dict[str, Any]]:
    return await asyncio.to_thread(get_thread_by_external_id, external_id)


async def add_message_async(thread_id: int, message: dict[str, Any]) -> int:
    return await asyncio.to_thread(add_message, thread_id, message)


async def get_messages_async(thread_id: int) -> list[dict[str, Any]]:
    return await asyncio.to_thread(get_messages, thread_id)

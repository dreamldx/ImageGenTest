"""SQLite connection and schema management for chat history persistence.

The DB file path defaults to ``data/chat_history.sqlite3`` and can be
overridden with the ``CHAT_HISTORY_DB_PATH`` environment variable.
"""

import os
import sqlite3
import threading
from pathlib import Path

import dotenv

from . import sql

dotenv.load_dotenv()

# Anchored to the project root (parent of this package), not the process's
# current working directory, so the DB always lands under <project root>/data
# regardless of where the bot is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chat_history.sqlite3"

DB_PATH = os.getenv("CHAT_HISTORY_DB_PATH", str(DEFAULT_DB_PATH))

# A single shared connection plus a lock: sqlite3 connections aren't safe
# for concurrent use across threads, and chat_history's async wrappers run
# on a thread-pool (`asyncio.to_thread`), so every access is serialized
# through `LOCK`.
LOCK = threading.Lock()
_connection: sqlite3.Connection | None = None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return the shared, lazily-initialized SQLite connection."""
    global _connection
    if _connection is None:
        _ensure_parent_dir(DB_PATH)
        connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute(sql.enable_foreign_keys())
        connection.execute(sql.enable_wal_mode())
        with LOCK:
            connection.executescript(sql.create_schema_script())
            connection.commit()
        _connection = connection
    return _connection

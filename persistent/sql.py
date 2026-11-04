"""Every SQL statement used by the `persistent` package, as functions.

Keeping them here (instead of inline in db.py / chat_history.py) means the
actual query text lives in one place and the calling code just reads as
"run this named query with these params".
"""


def create_schema_script() -> str:
    """Full schema: `threads` and `messages` as separate, linked tables."""
    return """
    CREATE TABLE IF NOT EXISTS threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        external_id TEXT UNIQUE NOT NULL,
        title TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );

    CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages (thread_id, id);
    """


def enable_foreign_keys() -> str:
    return "PRAGMA foreign_keys = ON"


def enable_wal_mode() -> str:
    """WAL journal mode: readers get a consistent shared-read snapshot and
    are never blocked by a concurrent writer (and vice versa), unlike the
    default rollback-journal mode, which locks the whole file on write."""
    return "PRAGMA journal_mode = WAL"


def select_thread_id_by_external_id() -> str:
    return "SELECT id FROM threads WHERE external_id = ?"


def select_thread_by_external_id() -> str:
    return "SELECT id, external_id, title, created_at FROM threads WHERE external_id = ?"


def insert_thread() -> str:
    return "INSERT INTO threads (external_id, title) VALUES (?, ?)"


def insert_message() -> str:
    return "INSERT INTO messages (thread_id, role, content, data) VALUES (?, ?, ?, ?)"


def select_messages_by_thread_id() -> str:
    return "SELECT data FROM messages WHERE thread_id = ? ORDER BY id ASC"

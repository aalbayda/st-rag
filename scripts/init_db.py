
from __future__ import annotations

import sqlite3
import sys

from app.config import get_settings
from app.db.engine import init_db


def main() -> None:
    settings = get_settings()
    db_path = settings.db_path

    print(f"Bootstrapping SQLite database at: {db_path}")

    init_db()

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        journal_mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()

    print(f"Journal mode: {journal_mode}")
    print(f"Tables created: {', '.join(tables) if tables else '(none)'}")

    expected = {"files", "chat_sessions", "messages"}
    missing = expected - set(tables)
    if missing:
        print(f"ERROR: missing tables: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    expected_journal = settings.db_journal_mode.lower()
    if journal_mode.lower() != expected_journal:
        print(
            f"ERROR: expected '{expected_journal}' journal mode, got '{journal_mode}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Database initialized successfully.")


if __name__ == "__main__":
    main

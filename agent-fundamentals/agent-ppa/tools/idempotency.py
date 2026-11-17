import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIRECTORY = PROJECT_ROOT / "data"
DATABASE_FILE = DATABASE_DIRECTORY / "idempotency.db"

class IdempotencyStore:

    def __init__(self, database_path: Path = DATABASE_FILE):
        self.database_path = database_path
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self._initialize_database()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=10
        )
        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    idempotency_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _hash_arguments(arguments: dict[str, Any]) -> str:
        canonical_arguments = json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False
        )
        return hashlib.sha256(
            canonical_arguments.encode("utf-8")
        ).hexdigest()
    
    def lookup(
        self,
        idempotency_key: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Check whether an operation already exists.

        When tool_name and arguments are provided, also verify
        that the key belongs to the same logical operation.
        """
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT tool_name, arguments_hash, status, result
                FROM operations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,)
            ).fetchone()

        if existing is None:
            return None

        if tool_name is not None and arguments is not None:
            arguments_hash = self._hash_arguments(arguments)

            if (
                existing["tool_name"] != tool_name
                or existing["arguments_hash"] != arguments_hash
            ):
                return {"status": "conflict", "result": None}

        return {
            "status": existing["status"],
            "result": existing["result"]
        }


    def claim(
        self,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        arguments_hash = self._hash_arguments(
            arguments=arguments
        )
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM operations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,)
            ).fetchone()

            if existing is not None:
                if (
                    existing["tool_name"] != tool_name
                    or existing["arguments_hash"] != arguments_hash
                ):
                    return {
                        "status": "conflict",
                        "result": None
                    }

                return {
                    "status": existing["status"],
                    "result": existing["result"]
                }

            connection.execute(
                """
                INSERT INTO operations (
                    idempotency_key,
                    tool_name,
                    arguments_hash,
                    status,
                    result,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    tool_name,
                    arguments_hash,
                    "in_progress",
                    None,
                    now,
                    now
                )
            )

            return {
                "status": "new",
                "result": None
            }

    def finish(
        self,
        idempotency_key: str,
        result: str,
        status: str = "completed"
    ) -> None:

        if status not in {"completed", "failed"}:
            raise ValueError("Invalid final operation status")

        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET status = ?,
                    result = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    status,
                    result,
                    now,
                    idempotency_key
                )
            )

idempotency_store = IdempotencyStore()

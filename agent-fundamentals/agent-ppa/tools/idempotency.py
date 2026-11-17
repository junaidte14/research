import hashlib
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import uuid

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
                    updated_at TEXT NOT NULL,
                    lease_expires_at TEXT,
                    lease_token TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0
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
        arguments: dict[str, Any],
        lease_duration_seconds: int = 60
    ) -> dict[str, Any]:

        if lease_duration_seconds <= 0:
            raise ValueError(
                "Lease duration must be greater than zero."
            )

        arguments_hash = self._hash_arguments(arguments)

        now = datetime.now(timezone.utc)

        lease_expires_at = (
            now + timedelta(seconds=lease_duration_seconds)
        ).isoformat()

        lease_token = str(uuid.uuid4())

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
                        "result": None,
                        "lease_token": None
                    }

                return {
                    "status": existing["status"],
                    "result": existing["result"],
                    "lease_token": None
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
                    updated_at,
                    lease_expires_at,
                    lease_token,
                    attempt_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    tool_name,
                    arguments_hash,
                    "in_progress",
                    None,
                    now.isoformat(),
                    now.isoformat(),
                    lease_expires_at,
                    lease_token,
                    1
                )
            )

            return {
                "status": "new",
                "result": None,
                "lease_token": lease_token
            }

    def finish(
        self,
        idempotency_key: str,
        lease_token: str,
        result: str,
        status: str = "completed"
    ) -> bool:
        if status not in {"completed", "failed"}:
            raise ValueError("Invalid final operation status.")

        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                SET status = ?,
                    result = ?,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    lease_token = NULL
                WHERE idempotency_key = ?
                AND lease_token = ?
                AND status = 'in_progress'
                AND lease_expires_at > ?
                """,
                (
                    status,
                    result,
                    now,
                    idempotency_key,
                    lease_token,
                    now
                )
            )

            return cursor.rowcount == 1

    def recover_expired_operations(self) -> int:
        """
        Mark expired in-progress operations as requiring recovery.
        Returns the number of operations updated.
        """

        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:

            cursor = connection.execute(
                """
                UPDATE operations
                SET status = 'recovery_required',
                    updated_at = ?,
                    lease_token = NULL,
                    lease_expires_at = NULL
                WHERE status = 'in_progress'
                AND lease_expires_at <= ?
                """,
                (now, now)
            )

            return cursor.rowcount

    def renew_lease(
        self,
        idempotency_key: str,
        lease_token: str,
        lease_duration_seconds: int = 60
    ) -> bool:
        """
        Extend an active operation's lease.

        Only the current lease owner can renew it.
        Expired leases cannot be renewed.
        """
        if lease_duration_seconds <= 0:
            raise ValueError("Lease duration must be greater than zero.")

        now = datetime.now(timezone.utc)
        new_expiry = (
            now + timedelta(seconds=lease_duration_seconds)
        ).isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                SET lease_expires_at = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                AND lease_token = ?
                AND status = 'in_progress'
                AND lease_expires_at > ?
                """,
                (
                    new_expiry,
                    now.isoformat(),
                    idempotency_key,
                    lease_token,
                    now.isoformat()
                )
            )

            return cursor.rowcount == 1

    def get_recovery_required_operations(self) -> list[dict[str, Any]]:
        """
        Return operations that require manual recovery or inspection.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    idempotency_key,
                    tool_name,
                    status,
                    result,
                    attempt_count,
                    created_at,
                    updated_at
                FROM operations
                WHERE status = 'recovery_required'
                ORDER BY updated_at ASC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def cancel(
        self,
        idempotency_key: str,
        lease_token: str,
        result: str = "Tool execution cancelled by user."
    ) -> bool:
        """
        Cancel an operation owned by the supplied lease token.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                SET status = 'cancelled',
                    result = ?,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    lease_token = NULL
                WHERE idempotency_key = ?
                AND lease_token = ?
                AND status = 'in_progress'
                AND lease_expires_at > ?
                """,
                (
                    result,
                    now,
                    idempotency_key,
                    lease_token,
                    now
                )
            )

            return cursor.rowcount == 1

idempotency_store = IdempotencyStore()

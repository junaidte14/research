import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIRECTORY = PROJECT_ROOT / "logs"
AUDIT_FILE = LOG_DIRECTORY / "audit.jsonl"

def write_audit_event(event: str, **details: Any) -> None:
    """
    Append one structured audit event to a JSONL file.
    """

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details
    }

    with AUDIT_FILE.open(
        mode="a",
        encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
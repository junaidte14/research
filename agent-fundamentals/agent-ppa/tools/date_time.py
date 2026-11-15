from datetime import datetime
import json

def get_current_datetime() -> str:
    """
    Returns the current local date and time,
    including timezone information.
    """

    now = datetime.now().astimezone()

    result = {
        "datetime": now.isoformat(),
        "timezone": str(now.tzinfo),
        "weekday": now.strftime("%A")
    }

    return json.dumps(result)
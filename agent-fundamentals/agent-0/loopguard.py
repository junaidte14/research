"""loopguard.py: detects an agent that is stuck repeating the same read-only call."""
import json

from safety import resolve


def normalize(arguments: str) -> str:
    """Make equivalent calls look identical: '../..' and '../../' are the same folder."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    if isinstance(args, dict):
        for key in ("path", "directory"):
            if isinstance(args.get(key), str):
                try:
                    args[key] = str(resolve(args[key]))
                except Exception:
                    pass
    return json.dumps(args, sort_keys=True)


class LoopGuard:
    # Tools that change state. Results seen earlier may be outdated afterwards, so counting restarts.
    MUTATING = {"write_file", "edit_file", "delete_file", "run_command", "remember"}

    def __init__(self, warn_at: int = 2, stop_at: int = 3):
        self.warn_at = warn_at
        self.stop_at = stop_at
        self.counts = {}

    def reset(self) -> None:
        """Forget all counts, for example after old tool results were pruned: re-fetching them is legitimate."""
        self.counts.clear()

    def check(self, name: str, arguments: str) -> str:
        """Return 'ok', 'warn' (2nd identical call) or 'stop' (3rd)."""
        if name in self.MUTATING:
            self.reset()            # re-reading a file after editing it is legitimate
            return "ok"
        key = (name, normalize(arguments))
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] >= self.stop_at:
            return "stop"
        if self.counts[key] >= self.warn_at:
            return "warn"
        return "ok"
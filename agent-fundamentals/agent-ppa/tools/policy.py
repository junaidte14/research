#Our first Policy Decision Point (PDP)

from dataclasses import dataclass

@dataclass(frozen=True)
class ToolPolicy:
    allowed_permissions: frozenset[str]

    def is_allowed(self, required_permission: str) -> bool:
        return required_permission in self.allowed_permissions

# Permissions available to this application
DEFAULT_TOOL_POLICY = ToolPolicy(
    allowed_permissions=frozenset({
        "math:calculate",
        "system:read_time",
        "text:analyze",
        "message:send"
    })
)
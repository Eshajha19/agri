from __future__ import annotations

import json
import logging
import threading
import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

KNOWN_ROLES = {"admin", "expert", "farmer", "vendor", "system", "guest"}


def validate_required_roles(required_roles: Iterable[str] | None) -> list[str] | None:
    if required_roles is None:
      return None

    normalized_roles: list[str] = []
    seen_roles: set[str] = set()

    for role in required_roles:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("RBAC required_roles must contain non-empty strings")

        normalized_role = role.strip().lower()
        if normalized_role not in KNOWN_ROLES:
            raise ValueError(f"Unknown RBAC role: {role}")

        if normalized_role not in seen_roles:
            seen_roles.add(normalized_role)
            normalized_roles.append(normalized_role)

    if not normalized_roles:
        raise ValueError("RBAC required_roles cannot be empty")

    return normalized_roles


@dataclass(slots=True)
class RBACAuditEvent:
    timestamp: str
    action: str
    path: str
    method: str
    outcome: str
    uid: str | None = None
    role: str | None = None
    required_roles: list[str] = field(default_factory=list)
    reason: str | None = None
    status_code: int | None = None
    source_ip: str | None = None


class RBACAuditTrail:
    def __init__(self, log_path: str | Path = "logs/rbac_audit.jsonl", max_events: int = 1000):
        self.log_path = Path(log_path)
        self.max_events = max_events
        self._events: deque[RBACAuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, event: RBACAuditEvent) -> RBACAuditEvent:
        payload = RBACAuditEvent(
            timestamp=event.timestamp,
            action=event.action,
            path=event.path,
            method=event.method,
            outcome=event.outcome,
            uid=event.uid,
            role=event.role,
            required_roles=list(event.required_roles or []),
            reason=event.reason,
            status_code=event.status_code,
            source_ip=event.source_ip,
        )

        async with self._lock:
            self._events.append(payload)
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(asdict(payload), ensure_ascii=True) + "\n")
            except OSError as exc:
                logger.warning("Unable to persist RBAC audit event: %s", exc)

        return payload

    def snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._lock:
            events = list(self._events)[-max(1, limit):]
        return [asdict(event) for event in events]

    def clear(self) -> None:
        async with self._lock:
            self._events.clear()


rbac_audit_trail = RBACAuditTrail()


def audit_rbac_event(
    *,
    request,
    action: str,
    outcome: str,
    uid: str | None = None,
    role: str | None = None,
    required_roles: Iterable[str] | None = None,
    reason: str | None = None,
    status_code: int | None = None,
) -> RBACAuditEvent:
    normalized_roles = validate_required_roles(required_roles)
    client_host = getattr(request.client, "host", None) if request else None
    return rbac_audit_trail.record(
        RBACAuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
            action=action,
            path=str(getattr(getattr(request, "url", None), "path", "unknown")),
            method=str(getattr(request, "method", "UNKNOWN")),
            outcome=outcome,
            uid=uid,
            role=role,
            required_roles=normalized_roles or [],
            reason=reason,
            status_code=status_code,
            source_ip=client_host,
        )
    )
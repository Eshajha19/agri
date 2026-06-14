"""GDPR-compliant deletion workflow with retention and audit trail."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


_TERMINAL_REQUEST_STATUSES = {"completed", "completed_with_errors"}


@dataclass(slots=True)
class DeletionTarget:
    """A user-scoped data sink that can be purged at deletion time."""

    name: str
    delete: Callable[[str], Any] | None = None
    retain_reason: str = ""


@dataclass(slots=True)
class GDPRDeletionRequest:
    request_id: str
    uid: str
    requested_by: str
    reason: str
    retention_days: int
    requested_at: str
    retention_until: str
    status: str = "pending_retention"
    completed_at: str = ""
    deleted_entities: list[str] = field(default_factory=list)
    retained_entities: list[str] = field(default_factory=list)
    target_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class GDPRAuditEvent:
    timestamp: str
    action: str
    uid: str
    request_id: str
    outcome: str
    details: dict[str, Any] = field(default_factory=dict)


class GDPRDeletionManager:
    """Retention-aware deletion registry with append-only audit logging."""

    def __init__(
        self,
        request_log_path: str | Path = "logs/gdpr_deletion_requests.jsonl",
        audit_log_path: str | Path = "logs/gdpr_deletion_audit.jsonl",
    ) -> None:
        self.request_log_path = Path(request_log_path)
        self.audit_log_path = Path(audit_log_path)
        self._request_lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._requests: dict[str, GDPRDeletionRequest] = {}
        self._post_deletion_hooks: list[Callable[[str], Any]] = []
        self._load_requests()

    @staticmethod
    def _should_keep_in_memory(request: GDPRDeletionRequest) -> bool:
        return request.status not in _TERMINAL_REQUEST_STATUSES
    def register_post_deletion_hook(self, callback_fn: Callable[[str], Any]) -> None:
        """Register a callback to be run after a deletion request is completed."""
        self._post_deletion_hooks.append(callback_fn)

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
        with lock:
            self._ensure_parent(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _load_requests(self) -> None:
        if not self.request_log_path.exists():
            return

        try:
            with self.request_log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    request = GDPRDeletionRequest(
                        request_id=payload["request_id"],
                        uid=payload["uid"],
                        requested_by=payload.get("requested_by", payload["uid"]),
                        reason=payload.get("reason", ""),
                        retention_days=int(payload.get("retention_days", 30)),
                        requested_at=payload["requested_at"],
                        retention_until=payload["retention_until"],
                        status=payload.get("status", "pending_retention"),
                        completed_at=payload.get("completed_at", ""),
                        deleted_entities=list(payload.get("deleted_entities", [])),
                        retained_entities=list(payload.get("retained_entities", [])),
                        target_results=list(payload.get("target_results", [])),
                    )
                    if self._should_keep_in_memory(request):
                        self._requests[request.request_id] = request
                    else:
                        self._requests.pop(request.request_id, None)
        except Exception as exc:
            logger.warning("Unable to load GDPR deletion requests: %s", exc)

    def _record_request(self, request: GDPRDeletionRequest) -> GDPRDeletionRequest:
        payload = asdict(request)
        self._append_jsonl(self.request_log_path, payload, self._request_lock)
        with self._request_lock:
            if self._should_keep_in_memory(request):
                self._requests[request.request_id] = request
            else:
                self._requests.pop(request.request_id, None)
        return request

    def _record_audit(self, event: GDPRAuditEvent) -> GDPRAuditEvent:
        self._append_jsonl(self.audit_log_path, asdict(event), self._audit_lock)
        return event

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _normalize_target_result(result: Any) -> dict[str, Any]:
        if result is None:
            return {"deleted": 0, "retained": 1, "notes": "no_action"}

        if isinstance(result, bool):
            return {"deleted": int(result), "retained": int(not result), "notes": "bool_result"}

        if isinstance(result, int):
            return {"deleted": max(0, result), "retained": 0 if result else 1, "notes": "int_result"}

        if isinstance(result, dict):
            deleted = int(result.get("deleted", result.get("deleted_count", 0)) or 0)
            retained = int(result.get("retained", result.get("retained_count", 0)) or 0)
            notes = str(result.get("notes", result.get("reason", "")))
            return {"deleted": deleted, "retained": retained, "notes": notes}

        return {"deleted": 0, "retained": 1, "notes": str(result)}

    def create_request(
        self,
        uid: str,
        *,
        requested_by: str | None = None,
        reason: str = "user_requested_erasure",
        retention_days: int = 30,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not uid or not uid.strip():
            raise ValueError("uid is required")

        retention_days = max(0, int(retention_days))
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        request = GDPRDeletionRequest(
            request_id=f"gdpr-{uuid.uuid4().hex[:12]}",
            uid=uid.strip(),
            requested_by=(requested_by or uid).strip(),
            reason=reason.strip() if reason else "user_requested_erasure",
            retention_days=retention_days,
            requested_at=current_time.isoformat(),
            retention_until=(current_time + timedelta(days=retention_days)).isoformat(),
        )
        self._record_request(request)
        self._record_audit(
            GDPRAuditEvent(
                timestamp=current_time.isoformat(),
                action="deletion_requested",
                uid=request.uid,
                request_id=request.request_id,
                outcome="accepted",
                details={
                    "retention_days": retention_days,
                    "retention_until": request.retention_until,
                    "reason": request.reason,
                },
            )
        )
        return asdict(request)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._request_lock:
            request = self._requests.get(request_id)
            return asdict(request) if request else None

    def list_requests(self, uid: str | None = None) -> list[dict[str, Any]]:
        with self._request_lock:
            requests = list(self._requests.values())
        if uid is not None:
            requests = [request for request in requests if request.uid == uid]
        requests.sort(key=lambda request: request.requested_at)
        return [asdict(request) for request in requests]

    def due_requests(self, now: datetime | None = None) -> list[dict[str, Any]]:
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._request_lock:
            requests = [
                request
                for request in self._requests.values()
                if request.status == "pending_retention"
                and self._parse_datetime(request.retention_until) <= current_time
            ]
        requests.sort(key=lambda request: request.requested_at)
        return [asdict(request) for request in requests]

    def execute_request(
        self,
        request_id: str,
        targets: Iterable[DeletionTarget],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._request_lock:
            request = self._requests.get(request_id)
            if request is None:
                raise KeyError(f"Unknown GDPR deletion request: {request_id}")
            if request.status != "pending_retention":
                return asdict(request)
            if self._parse_datetime(request.retention_until) > current_time:
                raise ValueError("Retention period has not elapsed yet")

        target_results: list[dict[str, Any]] = []
        deleted_entities: list[str] = []
        retained_entities: list[str] = []
        errors: list[str] = []

        for target in targets:
            try:
                normalized = self._normalize_target_result(target.delete(request.uid) if target.delete else None)
                result = {
                    "target": target.name,
                    **normalized,
                }
                target_results.append(result)
                if normalized["deleted"] > 0:
                    deleted_entities.append(target.name)
                else:
                    retained_entities.append(target.name)
            except Exception as exc:
                error_message = f"{target.name}: {exc}"
                target_results.append({"target": target.name, "deleted": 0, "retained": 1, "notes": error_message})
                retained_entities.append(target.name)
                errors.append(error_message)

        with self._request_lock:
            request = self._requests[request_id]
            request.status = "completed_with_errors" if errors else "completed"
            request.completed_at = current_time.isoformat()
            request.deleted_entities = deleted_entities
            request.retained_entities = retained_entities
            request.target_results = target_results
            self._record_request(request)

        # Trigger registered post-deletion callbacks
        for hook in self._post_deletion_hooks:
            try:
                hook(request.uid)
            except Exception as exc:
                logger.error("Error executing post-deletion hook for user %s: %s", request.uid, exc)

        self._record_audit(
            GDPRAuditEvent(
                timestamp=current_time.isoformat(),
                action="deletion_completed",
                uid=request.uid,
                request_id=request.request_id,
                outcome=request.status,
                details={
                    "deleted_entities": deleted_entities,
                    "retained_entities": retained_entities,
                    "errors": errors,
                },
            )
        )
        return asdict(request)

    def process_due_requests(
            self,
            target_builder: Callable[[str], Iterable[DeletionTarget]],
            *,
            now: datetime | None = None,
        ) -> list[dict[str, Any]]:
            """
            Process only approved GDPR deletion requests that are due.
            target_builder: function that takes a uid and returns deletion targets for that user.
            """
            results = []
            for request in self.due_requests(now=now):
                uid = request["uid"]
                request_id = request["request_id"]
                reason = request["reason"]
                requested_by = request["requested_by"]

                # Build user-scoped targets instead of system-wide
                targets = target_builder(uid)

                result = self.execute_request(request_id, targets, now=now)

                # Add audit details for actor + reason
                self._record_audit(
                    GDPRAuditEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        action="gdpr_delete",
                        uid=uid,
                        request_id=request_id,
                        outcome=result["status"],
                        details={
                            "requested_by": requested_by,
                            "reason": reason,
                            "deleted_entities": result.get("deleted_entities", []),
                            "retained_entities": result.get("retained_entities", []),
                        },
                    )
                )
                results.append(result)
            return results

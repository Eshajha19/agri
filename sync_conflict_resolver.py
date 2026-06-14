"""
Sync Conflict Resolution Engine
Handles concurrent updates, version tracking, and conflict resolution
"""

import logging
import threading
from collections import deque
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json

CONFLICT_LOG_MAX_ENTRIES = 1000

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    SERVER_WINS = "server_wins"
    MERGE = "merge"
    CRDT_LWW = "crdt_lww"


class SyncState(Enum):
    SYNCED = "synced"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"
    OFFLINE = "offline"


class VersionVector:
    def __init__(self, vector: Dict[str, int] = None):
        self.vector = vector or {}

    def increment(self, client_id: str) -> None:
        self.vector[client_id] = self.vector.get(client_id, 0) + 1

    def merge(self, other: 'VersionVector') -> None:
        for client_id, version in other.vector.items():
            self.vector[client_id] = max(self.vector.get(client_id, 0), version)

    def happened_before(self, other: 'VersionVector') -> bool:
        """Check if this vector happened before another (strict)"""
        if self.vector == other.vector:
            return False

        # An empty vector represents unknown state — treat as concurrent.
        if not self.vector or not other.vector:
            return False

        at_least_one_less = False

        for client_id in set(list(self.vector.keys()) + list(other.vector.keys())):
            self_ver = self.vector.get(client_id, 0)
            other_ver = other.vector.get(client_id, 0)

            if self_ver > other_ver:
                return False
            if self_ver < other_ver:
                at_least_one_less = True

        return at_least_one_less

    def concurrent_with(self, other: 'VersionVector') -> bool:
        if self.vector == other.vector:
            return False
        return (not self.happened_before(other) and not other.happened_before(self))

    def to_dict(self) -> Dict[str, int]:
        return self.vector.copy()

    @staticmethod
    def from_dict(data: Dict[str, int]) -> 'VersionVector':
        return VersionVector(data.copy())


class DocumentVersion:
    def __init__(
        self,
        doc_id: str,
        data: Dict[str, Any],
        client_id: str,
        version_vector: VersionVector = None,
        timestamp: str = None,
        checksum: str = None,
        crdt_state: Dict[str, Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.data = data
        self.client_id = client_id
        self.version_vector = version_vector or VersionVector()
        self.timestamp = timestamp or datetime.now().isoformat()
        self.crdt_state = crdt_state or {
            key: {
                "value": value,
                "timestamp": self.timestamp,
                "client_id": self.client_id,
            }
            for key, value in data.items()
        }
        self.checksum = checksum or self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "data": self.data,
            "client_id": self.client_id,
            "version_vector": self.version_vector.to_dict(),
            "timestamp": self.timestamp,
            "checksum": self.checksum,
            "crdt_state": self.crdt_state,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'DocumentVersion':
        return DocumentVersion(
            doc_id=data["doc_id"],
            data=data["data"],
            client_id=data["client_id"],
            version_vector=VersionVector.from_dict(data.get("version_vector", {})),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum"),
            crdt_state=data.get("crdt_state")
        )


class ConflictDetector:
    @staticmethod
    def detect_conflict(
        local_version: DocumentVersion,
        server_version: DocumentVersion
    ) -> bool:
        if local_version.checksum == server_version.checksum:
            return False
        if local_version.version_vector.vector == server_version.version_vector.vector:
            return True
        if local_version.version_vector.concurrent_with(server_version.version_vector):
            return True
        return False

    @staticmethod
    def find_conflicting_fields(
        local_data: Dict[str, Any],
        server_data: Dict[str, Any],
        base_data: Dict[str, Any] = None
    ) -> List[str]:
        conflicting_fields = []
        all_keys = set(
            list(local_data.keys()) +
            list(server_data.keys()) +
            list((base_data or {}).keys())
        )
        for key in all_keys:
            local_val = local_data.get(key)
            server_val = server_data.get(key)
            if base_data:
                base_val = base_data.get(key)
                local_changed = local_val != base_val
                server_changed = server_val != base_val
                if local_changed and server_changed and local_val != server_val:
                    conflicting_fields.append(key)
            else:
                if key in local_data and key in server_data and local_val != server_val:
                    conflicting_fields.append(key)
        return conflicting_fields


class ConflictResolver:
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS):
        self.strategy = strategy
        self.conflict_log: deque = deque(maxlen=CONFLICT_LOG_MAX_ENTRIES)

    def resolve(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, bool, List[str]]:
        if self.strategy in (ConflictResolutionStrategy.MERGE, ConflictResolutionStrategy.CRDT_LWW):
            merged_version, conflicting_fields = self._apply_strategy(local_version, server_version, base_version)
            has_conflict = len(conflicting_fields) > 0
        else:
            has_conflict = ConflictDetector.detect_conflict(local_version, server_version)
            if not has_conflict:
                merged_version = server_version
                conflicting_fields = []
            else:
                merged_version, conflicting_fields = self._apply_strategy(local_version, server_version, base_version)
        return merged_version, has_conflict, conflicting_fields

    def _apply_strategy(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, List[str]]:
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return self._last_write_wins(local_version, server_version)
        elif self.strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return self._first_write_wins(local_version, server_version)
        elif self.strategy == ConflictResolutionStrategy.SERVER_WINS:
            return self._server_wins(local_version, server_version)
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            return self._three_way_merge(local_version, server_version, base_version)
        elif self.strategy == ConflictResolutionStrategy.CRDT_LWW:
            merged_version, conflicting_fields = CRDTResolver.merge(local_version, server_version)
            self._log_conflict("crdt_lww", local_version, server_version, merged_version)
            return merged_version, conflicting_fields
        else:
            return self._last_write_wins(local_version, server_version)
    
    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        """Parse an ISO-format timestamp string to datetime for comparison."""
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)

    def _last_write_wins(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion
    ) -> Tuple[DocumentVersion, List[str]]:
        """Use version with most recent timestamp"""
        conflicting_fields = ConflictDetector.find_conflicting_fields(
            local_version.data,
            server_version.data
        )
        
        if self._parse_ts(local_version.timestamp) > self._parse_ts(server_version.timestamp):
            winner = local_version
        else:
            winner = server_version
        
        self._log_conflict("last_write_wins", local_version, server_version, winner)
        return winner, conflicting_fields
    
    def _first_write_wins(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion
    ) -> Tuple[DocumentVersion, List[str]]:
        """Use version with earliest timestamp"""
        conflicting_fields = ConflictDetector.find_conflicting_fields(
            local_version.data,
            server_version.data
        )
        
        if self._parse_ts(local_version.timestamp) < self._parse_ts(server_version.timestamp):
            winner = local_version
        else:
            winner = server_version
        
        self._log_conflict("first_write_wins", local_version, server_version, winner)
        return winner, conflicting_fields

    def _server_wins(self, local_version: DocumentVersion, server_version: DocumentVersion) -> Tuple[DocumentVersion, List[str]]:
        conflicting_fields = ConflictDetector.find_conflicting_fields(local_version.data, server_version.data)
        self._log_conflict("server_wins", local_version, server_version, server_version)
        return server_version, conflicting_fields

    def _three_way_merge(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, List[str]]:
        base_data = (base_version or DocumentVersion("", {}, "system")).data
        merged_data = base_data.copy()
        conflicting_fields = []
        all_keys = set(list(local_version.data.keys()) + list(server_version.data.keys()) + list(base_data.keys()))
        for key in all_keys:
            local_val = local_version.data.get(key)
            server_val = server_version.data.get(key)
            base_val = base_data.get(key)
            local_changed = local_val != base_val
            server_changed = server_val != base_val
            if local_changed and server_changed:
                if local_val == server_val:
                    merged_data[key] = local_val
                else:
                    merged_data[key] = server_val
                    conflicting_fields.append(key)
            elif local_changed:
                merged_data[key] = local_val
            elif server_changed:
                merged_data[key] = server_val
        
        # Create merged version with combined causal history
        merged_vector = VersionVector(local_version.version_vector.vector.copy())
        merged_vector.merge(server_version.version_vector)
        merged_version = DocumentVersion(
            doc_id=server_version.doc_id,
            data=merged_data,
            client_id="system",
            version_vector=merged_vector,
            timestamp=datetime.now().isoformat()
        )
        
        self._log_conflict("three_way_merge", local_version, server_version, merged_version)
        return merged_version, conflicting_fields

    def _log_conflict(self, strategy: str, local: DocumentVersion, server: DocumentVersion, resolved: DocumentVersion) -> None:
        self.conflict_log.append({
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "doc_id": local.doc_id,
            "local_client": local.client_id,
            "server_client": server.client_id,
            "resolved_client": resolved.client_id,
            "local_checksum": local.checksum,
            "server_checksum": server.checksum,
            "resolved_checksum": resolved.checksum
        })
        logger.info(f"Conflict resolved: doc={local.doc_id}, strategy={strategy}, winner={resolved.client_id}")

    def get_conflict_log(self) -> List[Dict]:
        return list(self.conflict_log)


class CRDTResolver:
    @staticmethod
    def _wrap_if_needed(data: Dict[str, Any], client_id: str, timestamp: str = None) -> Dict[str, Dict]:
        ts = timestamp or datetime.now().isoformat()
        wrapped = {}
        for k, v in data.items():
            if isinstance(v, dict) and 'value' in v and 'timestamp' in v and 'client_id' in v:
                wrapped[k] = v
            else:
                wrapped[k] = {'value': v, 'timestamp': ts, 'client_id': client_id}
        return wrapped

    @staticmethod
    def _state_map(version: DocumentVersion) -> Dict[str, Dict[str, Any]]:
        if getattr(version, "crdt_state", None):
            return {key: dict(value) for key, value in version.crdt_state.items()}
        return CRDTResolver._wrap_if_needed(version.data, version.client_id, version.timestamp)

    @staticmethod
    def _state_order_key(state: Dict[str, Any]) -> Tuple[str, str, str]:
        value_blob = json.dumps(state.get("value"), sort_keys=True, default=str)
        checksum = hashlib.sha256(value_blob.encode("utf-8")).hexdigest()
        return (str(state.get("timestamp", "")), str(state.get("client_id", "")), checksum)

    @staticmethod
    def _choose_state(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        return left if CRDTResolver._state_order_key(left) >= CRDTResolver._state_order_key(right) else right

    @staticmethod
    def merge(local: DocumentVersion, server: DocumentVersion) -> Tuple[DocumentVersion, List[str]]:
        local_fields = CRDTResolver._state_map(local)
        server_fields = CRDTResolver._state_map(server)
        merged_fields: Dict[str, Any] = {}
        merged_state: Dict[str, Dict[str, Any]] = {}
        conflicting: List[str] = []
        keys = set(list(local_fields.keys()) + list(server_fields.keys()))
        for k in keys:
            lf = local_fields.get(k)
            sf = server_fields.get(k)
            if lf is None:
                chosen = sf
            elif sf is None:
                chosen = lf
            else:
                chosen = CRDTResolver._choose_state(lf, sf)
                if lf["value"] != sf["value"]:
                    conflicting.append(k)
            merged_fields[k] = chosen["value"]
            merged_state[k] = dict(chosen)
        merged_version = DocumentVersion(doc_id=server.doc_id, data=merged_fields, client_id="system", timestamp=datetime.now().isoformat(), crdt_state=merged_state)
        return merged_version, conflicting


class SyncManager:
    def __init__(self, resolver: ConflictResolver = None, use_crdt: bool = True):
        self.resolver = resolver or ConflictResolver()
        self.use_crdt = use_crdt
        self.document_versions: Dict[str, DocumentVersion] = {}
        self.sync_state = SyncState.SYNCED
        self.pending_syncs: Dict[str, DocumentVersion] = {}

    def on_local_change(self, doc_id: str, data: Dict[str, Any], client_id: str) -> None:
        version = DocumentVersion(doc_id=doc_id, data=data, client_id=client_id, timestamp=datetime.now().isoformat())
        self.pending_syncs[doc_id] = version
        self.sync_state = SyncState.SYNCING

    def on_server_update(self, doc_id: str, server_data: Dict[str, Any], server_version_vector: Dict[str, int] = None) -> Tuple[Dict[str, Any], bool, List[str]]:
        server_version = DocumentVersion(doc_id=doc_id, data=server_data, client_id="server", version_vector=VersionVector.from_dict(server_version_vector or {}))
        local_version = self.pending_syncs.get(doc_id) or self.document_versions.get(doc_id)
        if local_version:
            base_version = self.document_versions.get(doc_id)
            if self.use_crdt:
                resolved, conflicting = CRDTResolver.merge(local_version, server_version)
                has_conflict = len(conflicting) > 0
            else:
                resolved, has_conflict, conflicting = self.resolver.resolve(local_version, server_version, base_version)
        else:
            resolved = server_version
            has_conflict = False
            conflicting = []
        self.document_versions[doc_id] = resolved
        if doc_id in self.pending_syncs:
            del self.pending_syncs[doc_id]
        if has_conflict:
            self.sync_state = SyncState.CONFLICT
        elif not self.pending_syncs:
            self.sync_state = SyncState.SYNCED
        return resolved.data, has_conflict, conflicting

    def get_pending_syncs(self) -> List[DocumentVersion]:
        return list(self.pending_syncs.values())

    def get_sync_status(self) -> Dict[str, Any]:
        return {
            "state": self.sync_state.value,
            "pending_documents": len(self.pending_syncs),
            "total_documents": len(self.document_versions),
            "conflict_count": len([log for log in self.resolver.get_conflict_log()])
        }


# Global sync manager instance
_sync_manager: Optional[SyncManager] = None
_sync_manager_lock = threading.Lock()


def get_sync_manager() -> SyncManager:
    """Get or create global sync manager — thread-safe singleton."""
    global _sync_manager
    if _sync_manager is None:
        with _sync_manager_lock:
            if _sync_manager is None:
                _sync_manager = SyncManager()
    return _sync_manager
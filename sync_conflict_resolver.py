"""
Sync Conflict Resolution Engine
Handles concurrent updates, version tracking, and conflict resolution
"""

import logging
import threading
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    SERVER_WINS = "server_wins"
    MERGE = "merge"


class SyncState(Enum):
    """Sync states for tracking synchronization progress"""
    SYNCED = "synced"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"
    OFFLINE = "offline"


class VersionVector:
    """
    Version Vector for tracking causality
    Helps determine if updates are concurrent or sequential
    """
    
    def __init__(self, vector: Dict[str, int] = None):
        """
        Initialize version vector
        
        Args:
            vector: Dict mapping client_id to version count
        """
        self.vector = vector or {}
    
    def increment(self, client_id: str) -> None:
        """Increment version for a client"""
        self.vector[client_id] = self.vector.get(client_id, 0) + 1
    
    def merge(self, other: 'VersionVector') -> None:
        """Merge with another version vector (takes max)"""
        for client_id, version in other.vector.items():
            self.vector[client_id] = max(
                self.vector.get(client_id, 0),
                version
            )
    
    def happened_before(self, other: 'VersionVector') -> bool:
        """Check if this vector happened before another"""
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
        """Check if vectors are concurrent (neither happened before)"""
        if self.vector == other.vector:
            return False
        return (not self.happened_before(other) and
                not other.happened_before(self))
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return self.vector.copy()
    
    @staticmethod
    def from_dict(data: Dict[str, int]) -> 'VersionVector':
        """Create from dictionary"""
        return VersionVector(data.copy())


class DocumentVersion:
    """
    Versioned document with metadata for conflict detection
    """
    
    def __init__(
        self,
        doc_id: str,
        data: Dict[str, Any],
        client_id: str,
        version_vector: VersionVector = None,
        timestamp: str = None,
        checksum: str = None
    ):
        self.doc_id = doc_id
        self.data = data
        self.client_id = client_id
        self.version_vector = version_vector or VersionVector()
        self.timestamp = timestamp or datetime.now().isoformat()
        self.checksum = checksum or self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of data (FIPS-compliant)"""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "data": self.data,
            "client_id": self.client_id,
            "version_vector": self.version_vector.to_dict(),
            "timestamp": self.timestamp,
            "checksum": self.checksum
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'DocumentVersion':
        """Create from dictionary"""
        return DocumentVersion(
            doc_id=data["doc_id"],
            data=data["data"],
            client_id=data["client_id"],
            version_vector=VersionVector.from_dict(data.get("version_vector", {})),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum")
        )


class ConflictDetector:
    """
    Detects conflicts between concurrent updates
    """
    
    @staticmethod
    def detect_conflict(
        local_version: DocumentVersion,
        server_version: DocumentVersion
    ) -> bool:
        """
        Detect if two versions have conflicting changes
        
        Returns:
            True if conflict detected, False otherwise
        """
        # Same checksum = no conflict
        if local_version.checksum == server_version.checksum:
            return False

        # Equal vectors with different checksums = divergence without clock
        # advancement → treat as conflict to avoid silent overwrite.
        if local_version.version_vector.vector == server_version.version_vector.vector:
            return True

        # Check if versions are concurrent
        if local_version.version_vector.concurrent_with(server_version.version_vector):
            return True

        # Sequential updates (not concurrent) = no conflict
        return False
    
    @staticmethod
    def find_conflicting_fields(
        local_data: Dict[str, Any],
        server_data: Dict[str, Any],
        base_data: Dict[str, Any] = None
    ) -> List[str]:
        """
        Find which fields have conflicting changes
        
        Args:
            local_data: Local/client version
            server_data: Server version
            base_data: Original version before both changes
        
        Returns:
            List of field names with conflicts
        """
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
    """
    Resolves conflicts between concurrent updates
    """
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS):
        self.strategy = strategy
        self.conflict_log: List[Dict] = []
    
    def resolve(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, bool, List[str]]:
        """
        Resolve conflict between two versions
        
        Args:
            local_version: Client's version
            server_version: Server's version
            base_version: Original version before changes
        
        Returns:
            Tuple of (resolved_version, has_conflict, conflicting_fields)
        """
        # Detect conflict
        has_conflict = ConflictDetector.detect_conflict(local_version, server_version)
        
        if not has_conflict:
            # No conflict, use server version (more authoritative)
            merged_version = server_version
            conflicting_fields = []
        else:
            # Conflict detected, apply resolution strategy
            merged_version, conflicting_fields = self._apply_strategy(
                local_version,
                server_version,
                base_version
            )
        
        return merged_version, has_conflict, conflicting_fields
    
    def _apply_strategy(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, List[str]]:
        """Apply conflict resolution strategy"""
        
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return self._last_write_wins(local_version, server_version)
        
        elif self.strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return self._first_write_wins(local_version, server_version)
        
        elif self.strategy == ConflictResolutionStrategy.SERVER_WINS:
            return self._server_wins(local_version, server_version)
        
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            return self._three_way_merge(local_version, server_version, base_version)
        
        else:
            return self._last_write_wins(local_version, server_version)
    
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
        
        if local_version.timestamp > server_version.timestamp:
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
        
        if local_version.timestamp < server_version.timestamp:
            winner = local_version
        else:
            winner = server_version
        
        self._log_conflict("first_write_wins", local_version, server_version, winner)
        return winner, conflicting_fields
    
    def _server_wins(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion
    ) -> Tuple[DocumentVersion, List[str]]:
        """Server version always wins"""
        conflicting_fields = ConflictDetector.find_conflicting_fields(
            local_version.data,
            server_version.data
        )
        
        self._log_conflict("server_wins", local_version, server_version, server_version)
        return server_version, conflicting_fields
    
    def _three_way_merge(
        self,
        local_version: DocumentVersion,
        server_version: DocumentVersion,
        base_version: DocumentVersion = None
    ) -> Tuple[DocumentVersion, List[str]]:
        """
        Three-way merge: combines non-conflicting changes.
        Uses base version to determine what changed.
        Ensures deleted fields (set to None or missing) are completely removed
        from the merged document instead of being retained with a null value.
        """
        base_data = (base_version or DocumentVersion("", {}, "system")).data
        merged_data = base_data.copy()
        conflicting_fields = []
        
        # Get all fields
        all_keys = set(
            list(local_version.data.keys()) + 
            list(server_version.data.keys()) +
            list(base_data.keys())
        )
        
        for key in all_keys:
            local_val = local_version.data.get(key)
            server_val = server_version.data.get(key)
            base_val = base_data.get(key)
            
            # Check what changed
            local_changed = local_val != base_val
            server_changed = server_val != base_val
            
            if local_changed and server_changed:
                # Both changed the field
                if local_val == server_val:
                    # Same value, use it (could be None/deleted)
                    if local_val is not None:
                        merged_data[key] = local_val
                    elif key in merged_data:
                        del merged_data[key]
                else:
                    # Conflicting change, use server version (could be None/deleted)
                    if server_val is not None:
                        merged_data[key] = server_val
                    elif key in merged_data:
                        del merged_data[key]
                    conflicting_fields.append(key)
            elif local_changed:
                # Only local changed
                if local_val is not None:
                    merged_data[key] = local_val
                elif key in merged_data:
                    del merged_data[key]
            elif server_changed:
                # Only server changed
                if server_val is not None:
                    merged_data[key] = server_val
                elif key in merged_data:
                    del merged_data[key]
            else:
                # Neither changed
                if base_val is None and key in merged_data:
                    del merged_data[key]
        
        # Create merged version
        merged_version = DocumentVersion(
            doc_id=server_version.doc_id,
            data=merged_data,
            client_id="system",
            timestamp=datetime.now().isoformat()
        )
        
        self._log_conflict("three_way_merge", local_version, server_version, merged_version)
        return merged_version, conflicting_fields
    
    def _log_conflict(
        self,
        strategy: str,
        local: DocumentVersion,
        server: DocumentVersion,
        resolved: DocumentVersion
    ) -> None:
        """Log conflict resolution for audit trail"""
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
        
        logger.info(
            f"Conflict resolved: doc={local.doc_id}, "
            f"strategy={strategy}, winner={resolved.client_id}"
        )
    
    def get_conflict_log(self) -> List[Dict]:
        """Get conflict resolution log"""
        return self.conflict_log.copy()


class SyncManager:
    """
    Manages overall synchronization including version tracking,
    conflict detection, and conflict resolution
    """
    
    def __init__(self, resolver: ConflictResolver = None):
        self.resolver = resolver or ConflictResolver()
        self.document_versions: Dict[str, DocumentVersion] = {}
        self.sync_state = SyncState.SYNCED
        self.pending_syncs: Dict[str, DocumentVersion] = {}
    
    def on_local_change(
        self,
        doc_id: str,
        data: Dict[str, Any],
        client_id: str
    ) -> None:
        """Record local change"""
        version = DocumentVersion(
            doc_id=doc_id,
            data=data,
            client_id=client_id
        )
        self.pending_syncs[doc_id] = version
        self.sync_state = SyncState.SYNCING
    
    def on_server_update(
        self,
        doc_id: str,
        server_data: Dict[str, Any],
        server_version_vector: Dict[str, int] = None
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Handle server update with conflict detection
        
        Returns:
            (final_data, has_conflict, conflicting_fields)
        """
        server_version = DocumentVersion(
            doc_id=doc_id,
            data=server_data,
            client_id="server",
            version_vector=VersionVector.from_dict(server_version_vector or {})
        )
        
        # Get local version if exists
        local_version = self.pending_syncs.get(doc_id) or self.document_versions.get(doc_id)
        
        if local_version:
            base_version = self.document_versions.get(doc_id)
            resolved, has_conflict, conflicting = self.resolver.resolve(
                local_version,
                server_version,
                base_version
            )
        else:
            resolved = server_version
            has_conflict = False
            conflicting = []
        
        # Update stored version
        self.document_versions[doc_id] = resolved
        
        # Remove from pending
        if doc_id in self.pending_syncs:
            del self.pending_syncs[doc_id]
        
        # Update sync state
        if has_conflict:
            self.sync_state = SyncState.CONFLICT
        elif not self.pending_syncs:
            self.sync_state = SyncState.SYNCED
        
        return resolved.data, has_conflict, conflicting
    
    def get_pending_syncs(self) -> List[DocumentVersion]:
        """Get documents pending synchronization"""
        return list(self.pending_syncs.values())
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            "state": self.sync_state.value,
            "pending_documents": len(self.pending_syncs),
            "total_documents": len(self.document_versions),
            "conflict_count": len([
                log for log in self.resolver.get_conflict_log()
            ])
        }


# Global sync manager instance and lock
_sync_manager: Optional[SyncManager] = None
_sync_manager_lock = threading.Lock()


def get_sync_manager() -> SyncManager:
    """Get or create global sync manager (thread-safe)"""
    global _sync_manager
    
    if _sync_manager is None:
        with _sync_manager_lock:
            if _sync_manager is None:
                _sync_manager = SyncManager()
    
    return _sync_manager

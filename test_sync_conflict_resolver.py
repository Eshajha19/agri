"""
Test suite for Sync Conflict Resolution
Verifies version tracking, conflict detection, and resolution
"""

import pytest
from sync_conflict_resolver import (
    VersionVector,
    DocumentVersion,
    ConflictDetector,
    ConflictResolver,
    ConflictResolutionStrategy,
    SyncManager
)


class TestVersionVector:
    """Test version vector for causality tracking"""
    
    def test_increment_version(self):
        """Version vector should increment correctly"""
        vv = VersionVector()
        vv.increment("client1")
        assert vv.vector["client1"] == 1
        
        vv.increment("client1")
        assert vv.vector["client1"] == 2
    
    def test_happened_before(self):
        """Check causality ordering"""
        vv1 = VersionVector({"client1": 1, "client2": 1})
        vv2 = VersionVector({"client1": 2, "client2": 1})
        
        # vv1 happened before vv2
        assert vv1.happened_before(vv2)
        assert not vv2.happened_before(vv1)
    
    def test_concurrent_vectors(self):
        """Detect concurrent updates"""
        vv1 = VersionVector({"client1": 1, "client2": 0})
        vv2 = VersionVector({"client1": 0, "client2": 1})
        
        # Both vectors are concurrent
        assert vv1.concurrent_with(vv2)
        assert vv2.concurrent_with(vv1)


class TestDocumentVersion:
    """Test versioned documents"""
    
    def test_checksum_calculation(self):
        """Checksum should be consistent"""
        doc1 = DocumentVersion("doc1", {"name": "John", "age": 30}, "client1")
        doc2 = DocumentVersion("doc1", {"name": "John", "age": 30}, "client1")
        
        assert doc1.checksum == doc2.checksum
    
    def test_checksum_changes_with_data(self):
        """Checksum should change when data changes"""
        doc1 = DocumentVersion("doc1", {"name": "John"}, "client1")
        doc2 = DocumentVersion("doc1", {"name": "Jane"}, "client1")
        
        assert doc1.checksum != doc2.checksum


class TestConflictDetector:
    """Test conflict detection"""
    
    def test_no_conflict_same_checksum(self):
        """No conflict when checksums match"""
        doc1 = DocumentVersion("doc1", {"name": "John"}, "client1")
        doc2 = DocumentVersion("doc1", {"name": "John"}, "server")
        
        has_conflict = ConflictDetector.detect_conflict(doc1, doc2)
        assert not has_conflict
    
    def test_conflict_detected_on_concurrent_changes(self):
        """Conflict detected when both sides change"""
        # Base version
        base = {"name": "John", "age": 30}
        
        # Local change
        local_data = {"name": "John Doe", "age": 30}
        local = DocumentVersion("doc1", local_data, "client1")
        
        # Server change
        server_data = {"name": "John", "age": 31}
        server = DocumentVersion("doc1", server_data, "server")
        
        # Mark as concurrent
        local.version_vector.increment("client1")
        server.version_vector.increment("server")
        
        has_conflict = ConflictDetector.detect_conflict(local, server)
        assert has_conflict


class TestConflictResolver:
    """Test conflict resolution"""
    
    def test_last_write_wins_strategy(self):
        """Last write should win"""
        resolver = ConflictResolver(ConflictResolutionStrategy.LAST_WRITE_WINS)
        
        # Local version (older)
        local = DocumentVersion("doc1", {"name": "John"}, "client1")
        
        # Server version (newer)
        import time
        time.sleep(0.01)
        server = DocumentVersion("doc1", {"name": "Jane"}, "server")
        
        resolved, has_conflict, conflicts = resolver.resolve(local, server)
        
        # Server version should win (it's newer)
        assert resolved.data["name"] == "Jane"
    
    def test_three_way_merge(self):
        """Three-way merge should combine non-conflicting changes"""
        resolver = ConflictResolver(ConflictResolutionStrategy.MERGE)
        
        # Base version
        base = DocumentVersion("doc1", {"name": "John", "age": 30, "city": "NYC"}, "base")
        
        # Local: changed name
        local = DocumentVersion("doc1", {"name": "John Doe", "age": 30, "city": "NYC"}, "client1")
        
        # Server: changed age
        server = DocumentVersion("doc1", {"name": "John", "age": 31, "city": "NYC"}, "server")
        
        resolved, has_conflict, conflicts = resolver.resolve(local, server, base)
        
        # Merged should have both changes
        assert resolved.data["name"] == "John Doe"  # Local change
        assert resolved.data["age"] == 31  # Server change
        assert resolved.data["city"] == "NYC"  # Unchanged

    def test_three_way_merge_handles_deleted_fields(self):
        """Three-way merge should completely remove deleted fields (set to None or missing)"""
        resolver = ConflictResolver(ConflictResolutionStrategy.MERGE)
        
        # Base version: has name, age, city, and status
        base = DocumentVersion("doc1", {"name": "John", "age": 30, "city": "NYC", "status": "active"}, "base")
        
        # Local: deleted city (missing key) and status (set to None)
        local = DocumentVersion("doc1", {"name": "John Doe", "age": 30}, "client1")
        
        # Server: deleted city (missing key) and status (set to None)
        server = DocumentVersion("doc1", {"name": "John", "age": 31}, "server")
        
        resolved, has_conflict, conflicts = resolver.resolve(local, server, base)
        
        # City and Status should be completely deleted, not retained with None/null values
        assert "city" not in resolved.data
        assert "status" not in resolved.data
        assert resolved.data["name"] == "John Doe"
        assert resolved.data["age"] == 31

    def test_server_wins_strategy(self):
        """Server wins strategy"""
        resolver = ConflictResolver(ConflictResolutionStrategy.SERVER_WINS)
        
        local = DocumentVersion("doc1", {"name": "John"}, "client1")
        server = DocumentVersion("doc1", {"name": "Jane"}, "server")
        
        resolved, has_conflict, conflicts = resolver.resolve(local, server)
        
        assert resolved.data["name"] == "Jane"
        assert resolved.client_id == "server"


class TestSyncManager:
    """Test overall sync management"""
    
    def test_record_local_change(self):
        """Record local optimistic update"""
        manager = SyncManager()
        
        manager.on_local_change("doc1", {"name": "John"}, "client1")
        
        assert manager.sync_state.value == "syncing"
        assert len(manager.pending_syncs) == 1
    
    def test_handle_server_update_no_conflict(self):
        """Handle server update without conflict"""
        manager = SyncManager()
        
        # No local change
        result = manager.on_server_update("doc1", {"name": "John"})
        
        data, has_conflict, conflicts = result
        assert not has_conflict
        assert data["name"] == "John"
    
    def test_get_pending_syncs(self):
        """Get pending synchronizations"""
        manager = SyncManager()
        
        manager.on_local_change("doc1", {"name": "John"}, "client1")
        manager.on_local_change("doc2", {"name": "Jane"}, "client1")
        
        pending = manager.get_pending_syncs()
        assert len(pending) == 2
    
    def test_sync_status(self):
        """Get sync status"""
        manager = SyncManager()
        
        manager.on_local_change("doc1", {"name": "John"}, "client1")
        
        status = manager.get_sync_status()
        
        assert status["state"] == "syncing"
        assert status["pending_documents"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

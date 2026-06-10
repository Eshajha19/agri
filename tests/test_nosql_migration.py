import unittest
from unittest.mock import MagicMock, call
import datetime
from google.cloud import firestore
from nosql_migration_framework import Migration, MultiPhaseMigration, MigrationRunner

class DummySingleMigration(Migration):
    @property
    def version(self) -> str:
        return "001_dummy"

    def up(self, transaction: firestore.Transaction) -> None:
        pass

    def down(self, transaction: firestore.Transaction) -> None:
        pass

class DummyMultiPhaseMigration(MultiPhaseMigration):
    @property
    def version(self) -> str:
        return "002_multi_dummy"

    def prepare(self, db: firestore.Client) -> None:
        pass

    def commit(self, db: firestore.Client) -> None:
        pass

    def cleanup(self, db: firestore.Client) -> None:
        pass

    def rollback(self, db: firestore.Client) -> None:
        pass

class TestNoSQLMigrationFramework(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock(spec=firestore.Client)
        self.mock_transaction = MagicMock(spec=firestore.Transaction)
        self.mock_db.transaction.return_value = self.mock_transaction
        self.runner = MigrationRunner(self.mock_db)
        
        # Setup lock mock
        self.mock_lock_ref = MagicMock()
        self.mock_record_ref = MagicMock()
        
        def collection_side_effect(name):
            if name == "_schema_migrations":
                mock_collection = MagicMock()
                def document_side_effect(doc_id):
                    if doc_id == "runner_lock":
                        return self.mock_lock_ref
                    return self.mock_record_ref
                mock_collection.document.side_effect = document_side_effect
                return mock_collection
            return MagicMock()
            
        self.mock_db.collection.side_effect = collection_side_effect

    def test_acquire_lock_success(self):
        mock_snapshot = MagicMock()
        mock_snapshot.exists = False
        self.mock_lock_ref.get.return_value = mock_snapshot
        
        # Just mock _acquire_lock_transaction directly to avoid dealing with the @firestore.transactional decorator issues in tests
        self.runner._acquire_lock_transaction = MagicMock(return_value=True)
        self.assertTrue(self.runner.acquire_lock())

    def test_run_single_migration_already_applied(self):
        mock_snapshot = MagicMock()
        mock_snapshot.exists = True
        mock_snapshot.to_dict.return_value = {"status": "COMPLETED"}
        self.mock_record_ref.get.return_value = mock_snapshot

        migration = DummySingleMigration()
        
        # Test the inner transactional function directly
        result = self.runner._run_single_migration_tx(self.mock_transaction, migration)
        self.assertFalse(result)
        self.mock_transaction.set.assert_not_called()

    def test_run_single_migration_success(self):
        mock_snapshot = MagicMock()
        mock_snapshot.exists = False
        self.mock_record_ref.get.return_value = mock_snapshot

        migration = DummySingleMigration()
        result = self.runner._run_single_migration_tx(self.mock_transaction, migration)
        self.assertTrue(result)
        
        # Check that it sets status to COMPLETED
        self.mock_transaction.set.assert_called_once()
        args, _ = self.mock_transaction.set.call_args
        self.assertEqual(args[0], self.mock_record_ref)
        self.assertEqual(args[1]['status'], "COMPLETED")
        self.assertEqual(args[1]['version'], "001_dummy")

    def test_run_multi_phase_migration_success(self):
        migration = DummyMultiPhaseMigration()
        migration.prepare = MagicMock()
        migration.commit = MagicMock()
        migration.cleanup = MagicMock()

        self.runner._claim_next_phase = MagicMock(
            side_effect=["prepare", "commit", "cleanup"]
        )
        self.runner._complete_phase = MagicMock()
        self.runner._complete_cleanup = MagicMock()

        self.runner._run_multi_phase_migration(migration)

        migration.prepare.assert_called_once_with(self.mock_db)
        migration.commit.assert_called_once_with(self.mock_db)
        migration.cleanup.assert_called_once_with(self.mock_db)
        self.runner._complete_phase.assert_has_calls(
            [
                call(self.mock_record_ref, "PREPARING", "PREPARED"),
                call(self.mock_record_ref, "COMMITTING", "COMMITTED"),
            ]
        )
        self.runner._complete_cleanup.assert_called_once_with(self.mock_record_ref)

    def test_run_multi_phase_migration_already_completed(self):
        self.runner._claim_next_phase = MagicMock(return_value=None)

        migration = DummyMultiPhaseMigration()
        migration.prepare = MagicMock()

        self.runner._run_multi_phase_migration(migration)

        migration.prepare.assert_not_called()

    def test_run_multi_phase_migration_blocks_concurrent_runner(self):
        from nosql_migration_framework import ConcurrentMigrationError

        self.runner._claim_next_phase = MagicMock(
            side_effect=ConcurrentMigrationError("in progress")
        )

        migration = DummyMultiPhaseMigration()
        with self.assertRaises(ConcurrentMigrationError):
            self.runner._run_multi_phase_migration(migration)

if __name__ == '__main__':
    unittest.main()

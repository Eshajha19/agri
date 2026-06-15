import abc
import datetime
import logging
from typing import List, Any
from google.cloud import firestore

logger = logging.getLogger(__name__)

class Migration(abc.ABC):
    """Base class for a standard, single-transaction schema migration."""
    
    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Unique version identifier, e.g., '0001_initial_schema'"""
        pass

    @abc.abstractmethod
    def up(self, transaction: firestore.Transaction) -> None:
        """Apply the migration."""
        pass

    @abc.abstractmethod
    def down(self, transaction: firestore.Transaction) -> None:
        """Revert the migration."""
        pass


class MultiPhaseMigration(abc.ABC):
    """
    Base class for large-scale or multi-phase schema migrations
    that exceed single-transaction limits (500 docs in Firestore)
    or require distributed PREPARE/COMMIT/CLEANUP phases.
    """
    @property
    @abc.abstractmethod
    def version(self) -> str:
        pass

    @abc.abstractmethod
    def prepare(self, db: firestore.Client) -> None:
        """
        Phase 1: Prepare the data. For example, query all documents
        that need migration and mark them with a 'migrating' flag using
        batched writes.
        """
        pass

    @abc.abstractmethod
    def commit(self, db: firestore.Client) -> None:
        """
        Phase 2: Perform the actual data transformations on the prepared
        documents using batched writes.
        """
        pass

    @abc.abstractmethod
    def cleanup(self, db: firestore.Client) -> None:
        """
        Phase 3: Remove any temporary flags or intermediate state.
        """
        pass

    @abc.abstractmethod
    def rollback(self, db: firestore.Client) -> None:
        """
        Rollback the migration if prepare or commit fails.
        """
        pass

class MigrationRunner:
    def __init__(self, db: firestore.Client):
        self.db = db
        self.migrations_collection = "_schema_migrations"
        self.lock_doc_id = "runner_lock"

    @firestore.transactional
    def _acquire_lock_transaction(self, transaction: firestore.Transaction) -> bool:
        lock_ref = self.db.collection(self.migrations_collection).document(self.lock_doc_id)
        snapshot = lock_ref.get(transaction=transaction)
        now = datetime.datetime.now(datetime.timezone.utc)

        if snapshot.exists:
            data = snapshot.to_dict()
            if data is not None:
                locked_at = data.get("locked_at")
                # If lock is held for > 15 minutes, assume dead and forcefully acquire
                if locked_at and (now - locked_at).total_seconds() < 900:
                    return False

        transaction.set(lock_ref, {"locked_at": now})
        return True

    def acquire_lock(self) -> bool:
        transaction = self.db.transaction()
        try:
            return self._acquire_lock_transaction(transaction)
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False

    def release_lock(self):
        lock_ref = self.db.collection(self.migrations_collection).document(self.lock_doc_id)
        lock_ref.delete()

    @firestore.transactional
    def _run_single_migration_tx(self, transaction: firestore.Transaction, migration: Migration):
        # Re-check if migration is applied inside transaction
        record_ref = self.db.collection(self.migrations_collection).document(migration.version)
        snapshot = record_ref.get(transaction=transaction)
        if snapshot.exists and snapshot.to_dict() and snapshot.to_dict().get("status") == "COMPLETED":
            logger.info(f"Migration {migration.version} already applied. Skipping.")
            return False

        logger.info(f"Running migration {migration.version} (UP)...")
        migration.up(transaction)
        
        # Mark as completed
        transaction.set(record_ref, {
            "version": migration.version,
            "status": "COMPLETED",
            "applied_at": firestore.SERVER_TIMESTAMP
        })
        return True

    def _run_multi_phase_migration(self, migration: MultiPhaseMigration):
        record_ref = self.db.collection(self.migrations_collection).document(migration.version)
        snapshot = record_ref.get()
        state = {}
        if snapshot.exists and snapshot.to_dict():
            state = snapshot.to_dict()
        status = state.get("status")
        current_phase = state.get("current_phase")

        if status == "COMPLETED":
            logger.info(f"Multi-phase migration {migration.version} already applied. Skipping.")
            return

        # Determine which phase to resume from based on the last attempted phase.
        #   None / "preparing" / "prepare"  → run prepare
        #   "prepared" / "committing"       → skip prepare, run commit
        #   "committed" / "cleaning_up"     → skip prepare & commit, run cleanup
        resume_from = current_phase or "prepare"

        try:
            if resume_from == "prepare":
                logger.info(f"Phase 1 (PREPARE): {migration.version}")
                record_ref.set({
                    "version": migration.version,
                    "status": "PREPARING",
                    "current_phase": "prepare",
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
                migration.prepare(self.db)
                record_ref.update({"status": "PREPARED", "current_phase": "prepared", "updated_at": firestore.SERVER_TIMESTAMP})

            if resume_from in ("prepare", "prepared", "committing"):
                logger.info(f"Phase 2 (COMMIT): {migration.version}")
                record_ref.update({"status": "COMMITTING", "current_phase": "committing", "updated_at": firestore.SERVER_TIMESTAMP})
                migration.commit(self.db)
                record_ref.update({"status": "COMMITTED", "current_phase": "committed", "updated_at": firestore.SERVER_TIMESTAMP})

            logger.info(f"Phase 3 (CLEANUP): {migration.version}")
            record_ref.update({"status": "CLEANING_UP", "current_phase": "cleaning_up", "updated_at": firestore.SERVER_TIMESTAMP})
            migration.cleanup(self.db)

            record_ref.update({
                "status": "COMPLETED",
                "current_phase": None,
                "applied_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            logger.info(f"Multi-phase migration {migration.version} completed successfully.")

        except Exception as e:
            logger.error(f"Error during multi-phase migration {migration.version}: {e}. Initiating rollback.")
            record_ref.update({"status": "ROLLING_BACK", "updated_at": firestore.SERVER_TIMESTAMP})
            try:
                migration.rollback(self.db)
                record_ref.update({"status": "FAILED", "current_phase": current_phase or "prepare", "error": str(e), "updated_at": firestore.SERVER_TIMESTAMP})
            except Exception as rollback_err:
                logger.error(f"Rollback failed for {migration.version}: {rollback_err}")
                record_ref.update({
                    "status": "ROLLBACK_FAILED",
                    "current_phase": current_phase or "prepare",
                    "error": str(e),
                    "rollback_error": str(rollback_err)
                })
            raise

    def run_migration(self, migration: Any):
        if isinstance(migration, Migration):
            tx = self.db.transaction()
            try:
                self._run_single_migration_tx(tx, migration)
            except Exception as e:
                logger.error(f"Single transaction migration {migration.version} failed: {e}")
                raise
        elif isinstance(migration, MultiPhaseMigration):
            self._run_multi_phase_migration(migration)
        else:
            raise ValueError(f"Unknown migration type: {type(migration)}")

    def run_all(self, migrations: List[Any]):
        if not self.acquire_lock():
            logger.warning("Could not acquire migration lock. Another process is likely running migrations.")
            return

        try:
            for migration in migrations:
                self.run_migration(migration)
        finally:
            self.release_lock()

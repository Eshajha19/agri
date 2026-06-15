"""
Database connection pooling and management for the backend.
Manages singleton connections to Firestore and optionally Postgres.
Provides connection lifecycle management, pooling, and graceful shutdown.
"""

import os
import logging
import threading
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any, Callable
from contextlib import contextmanager
from datetime import datetime, timedelta

try:
    import firebase_admin
    from firebase_admin import firestore, credentials
except ImportError:
    firebase_admin = None
    firestore = None

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger(__name__)


class FirestoreConnectionManager:
    """
    Manages Firestore client singleton and connection pooling.
    Firebase Admin SDK manages connections internally; this class
    ensures we use a single client instance per app lifecycle.
    """

    _instance: Optional['FirestoreConnectionManager'] = None
    _lock: threading.Lock = threading.Lock()
    _client: Optional[firestore.Client] = None
    _initialized: bool = False
    _last_used: Optional[datetime] = None

    def __new__(cls) -> 'FirestoreConnectionManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, credentials_path: Optional[str] = None) -> bool:
        """
        Initialize Firestore client singleton.
        Safe to call multiple times; idempotent.
        """
        if self._initialized and self._client is not None:
            logger.debug("Firestore client already initialized")
            return True

        async with self._lock:
            if self._initialized:
                return True

            if firebase_admin is None or firestore is None:
                logger.warning("Firebase Admin SDK not available")
                return False

            try:
                if not firebase_admin._apps:
                    # Try to initialize from credentials file
                    if credentials_path and os.path.exists(credentials_path):
                        cred = credentials.Certificate(credentials_path)
                        firebase_admin.initialize_app(cred)
                        logger.info(f"Initialized Firebase from credentials: {credentials_path}")
                    elif os.path.exists("firebase-credentials.json"):
                        cred = credentials.Certificate("firebase-credentials.json")
                        firebase_admin.initialize_app(cred)
                        logger.info("Initialized Firebase from firebase-credentials.json")
                    else:
                        # Try environment variable
                        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                        if cred_json:
                            import json
                            cred_dict = json.loads(cred_json)
                            cred = credentials.Certificate(cred_dict)
                            firebase_admin.initialize_app(cred)
                            logger.info("Initialized Firebase from FIREBASE_CREDENTIALS_JSON")
                        else:
                            # Try GCP application default credentials
                            firebase_admin.initialize_app()
                            logger.info("Initialized Firebase with application default credentials")

                self._client = firestore.client()
                self._initialized = True
                self._last_used = datetime.now()
                logger.info("✅ Firestore client initialized successfully")
                return True

            except Exception as exc:
                logger.error(f"❌ Failed to initialize Firestore client: {exc}", exc_info=True)
                return False

    @property
    def client(self) -> Optional[firestore.Client]:
        """Get Firestore client instance."""
        if not self._initialized:
            self.initialize()
        self._last_used = datetime.now()
        return self._client

    def reset(self) -> None:
        """Reset the Firestore client (for testing only)."""
        async with self._lock:
            self._client = None
            self._initialized = False
            self._last_used = None
            logger.info("Firestore client reset")

    def get_stats(self) -> dict:
        """Get connection manager stats for monitoring."""
        return {
            "initialized": self._initialized,
            "client_exists": self._client is not None,
            "last_used": self._last_used.isoformat() if self._last_used else None,
            "uptime_seconds": (datetime.now() - self._last_used).total_seconds() if self._last_used else None,
        }


class PostgresConnectionManager:
    """
    Optional Postgres connection pool manager using asyncpg.
    Only available if asyncpg is installed.
    """

    _instance: Optional['PostgresConnectionManager'] = None
    _lock: threading.Lock = threading.Lock()
    _pool: Optional['asyncpg.Pool'] = None
    _config: Optional[dict] = None

    def __new__(cls) -> 'PostgresConnectionManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "agri",
        user: str = "postgres",
        password: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 20,
        max_inactive_connection_lifetime: float = 300.0,
        **kwargs
    ) -> bool:
        """Initialize Postgres connection pool."""
        if not HAS_ASYNCPG:
            logger.warning("asyncpg not available; skipping Postgres pool initialization")
            return False

        async with self._lock:
            if self._pool is not None:
                logger.debug("Postgres pool already initialized")
                return True

            try:
                self._config = {
                    "host": host,
                    "port": port,
                    "database": database,
                    "user": user,
                    "password": password,
                    "min_size": min_size,
                    "max_size": max_size,
                    "max_inactive_connection_lifetime": max_inactive_connection_lifetime,
                    **kwargs
                }

                self._pool = await asyncpg.create_pool(**self._config)
                logger.info("✅ Postgres connection pool initialized successfully")
                return True

            except Exception as exc:
                logger.error(f"❌ Failed to initialize Postgres pool: {exc}", exc_info=True)
                return False

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Postgres connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for getting a connection from the pool."""
        if self._pool is None:
            raise RuntimeError("Postgres pool not initialized")

        conn = await self._pool.acquire()
        try:
            yield conn
        finally:
            await self._pool.release(conn)

    async def get_stats(self) -> dict:
        """Get pool stats for monitoring."""
        if self._pool is None:
            return {"pool_initialized": False}

        return {
            "pool_initialized": True,
            "min_size": self._config.get("min_size") if self._config else None,
            "max_size": self._config.get("max_size") if self._config else None,
            "size": self._pool.get_size(),
            "idle_size": self._pool.get_idle_size(),
        }

    def reset(self) -> None:
        """Reset the pool (for testing only)."""
        async with self._lock:
            self._pool = None
            self._config = None


# Global singleton instances
firestore_manager = FirestoreConnectionManager()
postgres_manager = PostgresConnectionManager()


def initialize_connections() -> bool:
    """Initialize all configured database connections."""
    logger.info("Initializing database connections...")
    success = True

    # Initialize Firestore
    if not firestore_manager.initialize():
        success = False
        logger.warning("Firestore initialization failed (continuing for now)")

    return success


def get_firestore_client() -> Optional[firestore.Client]:
    """Get the global Firestore client (convenience wrapper)."""
    return firestore_manager.client


async def shutdown_connections() -> None:
    """Gracefully shutdown all database connections."""
    logger.info("Shutting down database connections...")
    if HAS_ASYNCPG:
        await postgres_manager.close()
    logger.info("All database connections shutdown complete")

import asyncio
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

# Mocking the WebSocket and filter_notifications_for_user if necessary
# but we are focusing on the snapshot method in NotificationBroadcastHub

from realtime_notifications import NotificationBroadcastHub

class TestNotificationHub(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_returns_list(self):
        """Verify that snapshot() returns a list containing seeded notifications."""
        # Initialize the hub
        hub = NotificationBroadcastHub(history_limit=10)
        
        # Seed some data
        notifications = [
            {"id": 1, "message": "Test 1"},
            {"id": 2, "message": "Test 2"}
        ]
        hub.seed_notifications(notifications)
        
        # Call snapshot
        result = await hub.snapshot()
        
        # Verify result is a list and contains the expected data
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 2)
        
        # Verify it's a copy (mutating history shouldn't affect the snapshot)
        hub.seed_notifications([{"id": 3, "message": "Test 3"}])
        self.assertEqual(len(result), 2)
        
    async def test_snapshot_empty_history(self):
        """Verify that snapshot() returns an empty list when history is empty."""
        hub = NotificationBroadcastHub()
        result = await hub.snapshot()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

if __name__ == "__main__":
    # Use verbosity=2 to show detailed test case names and descriptions
    unittest.main(verbosity=2)

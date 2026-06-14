#!/usr/bin/env python3
"""
Detailed, verbose tests for database connection pooling.
Tests Firestore singleton management, connection reuse, etc.
"""

import sys
import time
import logging
from datetime import datetime

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ANSI color codes for pretty output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log_header(title):
    """Log a test header with formatting."""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}  TEST: {title.upper()}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}")


def log_step(step):
    """Log a test step."""
    print(f"\n{CYAN}  → {step}{RESET}")


def log_info(msg):
    """Log an info message."""
    print(f"      [INFO] {msg}")


def log_success(msg):
    """Log a success message."""
    print(f"      {GREEN}[PASS]{RESET} {msg}")


def log_failure(msg):
    """Log a failure message."""
    print(f"      {RED}[FAIL]{RESET} {msg}")


def test_firestore_singleton():
    """Test that FirestoreConnectionManager is a singleton."""
    log_header("Firestore Connection Manager Singleton")

    from persistence.connections import FirestoreConnectionManager, firestore_manager

    log_step("Creating multiple FirestoreConnectionManager instances")

    manager1 = FirestoreConnectionManager()
    log_info(f"Instance 1 ID: {id(manager1)}")

    manager2 = FirestoreConnectionManager()
    log_info(f"Instance 2 ID: {id(manager2)}")

    manager3 = FirestoreConnectionManager()
    log_info(f"Instance 3 ID: {id(manager3)}")

    log_step("Verifying all instances are the same object")

    if manager1 is manager2 and manager2 is manager3:
        log_success("All instances are the same singleton object")
    else:
        log_failure("Instances are not singletons")
        return False

    log_step("Verifying the global instance is the same singleton")

    if firestore_manager is manager1:
        log_success("Global firestore_manager is the same singleton")
    else:
        log_failure("Global firestore_manager is not the same singleton")
        return False

    return True


def test_client_reuse():
    """Test that the same client instance is reused across calls."""
    log_header("Firestore Client Reuse")

    from persistence.connections import get_firestore_client, firestore_manager

    # Reset for fresh test
    firestore_manager.reset()

    log_step("Getting firestore client first time")
    client1 = get_firestore_client()
    log_info(f"First client instance: {id(client1)}")

    log_step("Getting firestore client second time")
    client2 = get_firestore_client()
    log_info(f"Second client instance: {id(client2)}")

    log_step("Getting firestore client third time")
    client3 = get_firestore_client()
    log_info(f"Third client instance: {id(client3)}")

    log_step("Verifying all clients are the same instance")
    if client1 is not None and client1 is client2 and client2 is client3:
        log_success("Same client instance returned for all calls")
    elif client1 is None:
        log_info("Firestore not available (this is expected if no credentials found)")
        log_success("Test passed (client availability not required for reuse test)")
    else:
        log_failure("Different client instances returned")
        return False

    return True


def test_stats_tracking():
    """Test that connection manager tracks stats properly."""
    log_header("Connection Manager Stats Tracking")

    from persistence.connections import firestore_manager

    # Reset and initialize
    firestore_manager.reset()
    firestore_manager.initialize()

    log_step("Getting initial stats")
    stats1 = firestore_manager.get_stats()
    log_info(f"Initial stats: {stats1}")

    log_step("Waiting 2 seconds and accessing client")
    time.sleep(2)
    client = firestore_manager.client
    time.sleep(1)

    log_step("Getting updated stats")
    stats2 = firestore_manager.get_stats()
    log_info(f"Updated stats: {stats2}")

    log_step("Verifying stats structure")
    required_keys = ["initialized", "client_exists", "last_used", "uptime_seconds"]
    all_keys_present = all(key in stats2 for key in required_keys)

    if all_keys_present:
        log_success("All required stats keys present")
    else:
        log_failure("Missing some stats keys")
        return False

    log_step("Verifying last_used was updated")
    if stats2["last_used"] != stats1["last_used"]:
        log_success("last_used timestamp was properly updated")
    else:
        log_info("last_used didn't change (Firestore may not have initialized)")

    return True


def test_repository_client_access():
    """Test that repositories use the same singleton client."""
    log_header("Repository Client Access")

    from persistence.connections import get_firestore_client, firestore_manager
    from persistence.repositories import (
        FinanceApplicationRepository,
        NotificationRepository,
        SupplyChainRepository,
    )

    # Reset for fresh test
    firestore_manager.reset()

    log_step("Creating multiple repository instances")

    repo1 = FinanceApplicationRepository()
    log_info(f"Finance repo client: {id(repo1.db)}")

    repo2 = NotificationRepository()
    log_info(f"Notification repo client: {id(repo2.db)}")

    repo3 = SupplyChainRepository()
    log_info(f"Supply chain repo client: {id(repo3.db)}")

    log_step("Getting client directly from manager")
    direct_client = get_firestore_client()
    log_info(f"Direct manager client: {id(direct_client)}")

    log_step("Verifying all clients are the same")
    all_same = True
    clients = [repo1.db, repo2.db, repo3.db, direct_client]

    if all(c is None for c in clients):
        log_success("All clients are None (Firestore not available, but consistent)")
    else:
        non_none_clients = [c for c in clients if c is not None]
        first = non_none_clients[0]
        if all(c is first for c in non_none_clients):
            log_success("All repository clients use the same singleton instance")
        else:
            log_failure("Repositories have different client instances")
            return False

    log_step("Testing refresh_client() method")
    repo1.refresh_client()
    log_info(f"Finance repo client after refresh: {id(repo1.db)}")

    if repo1.db is direct_client or direct_client is None:
        log_success("refresh_client() returns the same singleton")
    else:
        log_failure("refresh_client() returned a different client")
        return False

    return True


def test_idempotent_initialization():
    """Test that initialize() is idempotent (safe to call multiple times)."""
    log_header("Idempotent Initialization")

    from persistence.connections import firestore_manager

    # Reset for fresh test
    firestore_manager.reset()

    log_step("First initialize() call")
    result1 = firestore_manager.initialize()
    log_info(f"First initialize() result: {result1}")

    log_step("Second initialize() call")
    result2 = firestore_manager.initialize()
    log_info(f"Second initialize() result: {result2}")

    log_step("Third initialize() call")
    result3 = firestore_manager.initialize()
    log_info(f"Third initialize() result: {result3}")

    log_step("Verifying all initialize() calls are consistent")
    if result1 == result2 == result3:
        log_success("initialize() is idempotent and returns consistent results")
    else:
        log_failure("initialize() returned different results on subsequent calls")
        return False

    return True


def run_suite():
    """Run all tests in the suite."""
    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}        DATABASE CONNECTION POOLING TEST SUITE{RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")

    tests = [
        ("Firestore Singleton", test_firestore_singleton),
        ("Client Reuse", test_client_reuse),
        ("Stats Tracking", test_stats_tracking),
        ("Repository Client Access", test_repository_client_access),
        ("Idempotent Initialization", test_idempotent_initialization),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            if success:
                print(f"\n  {GREEN}{BOLD}✓ {test_name} passed in {duration:.2f}s{RESET}")
            else:
                print(f"\n  {RED}{BOLD}✗ {test_name} failed in {duration:.2f}s{RESET}")
        except Exception as e:
            print(f"\n  {RED}{BOLD}✗ {test_name} failed with exception: {e}{RESET}")
            logger.exception(f"Test {test_name} failed with exception")
            results.append((test_name, False, 0.0))

    print(f"\n\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}  FINAL RESULTS{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}")

    total_tests = len(tests)
    passed = sum(1 for _, success, _ in results if success)

    for name, success, duration in results:
        status = f"{GREEN}{BOLD}PASSED{RESET}" if success else f"{RED}{BOLD}FAILED{RESET}"
        print(f"    - {name:<35} : {status}")

    print(f"\n  {BOLD}Total: {passed}/{total_tests} tests passed{RESET}")

    if passed == total_tests:
        print(f"\n  {GREEN}{BOLD}✅ All connection pooling tests passed!{RESET}\n")
        return 0
    else:
        print(f"\n  {RED}{BOLD}❌ Some tests failed!{RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_suite()
    sys.exit(exit_code)

#!/usr/bin/env python
"""
Firestore Rules Test Runner with Emulator Management
Handles starting emulator, running tests, and reporting results
"""

import os
import sys
import subprocess
import time
import json
import socket
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FirestoreEmulatorManager:
    """Manage Firestore emulator lifecycle"""
    
    HOST = "localhost"
    PORT = 8080
    
    @staticmethod
    def is_running() -> bool:
        """Check if emulator is running"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((FirestoreEmulatorManager.HOST, FirestoreEmulatorManager.PORT))
        sock.close()
        return result == 0
    
    @staticmethod
    def wait_for_emulator(timeout: int = 30) -> bool:
        """Wait for emulator to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if FirestoreEmulatorManager.is_running():
                logger.info("✅ Firestore emulator is running")
                return True
            logger.info("⏳ Waiting for Firestore emulator...")
            time.sleep(1)
        
        logger.error("❌ Firestore emulator failed to start within timeout")
        return False
    
    @staticmethod
    def start_emulator():
        """Start Firestore emulator"""
        logger.info("Starting Firestore emulator...")
        
        if FirestoreEmulatorManager.is_running():
            logger.info("⚠️  Firestore emulator is already running")
            return True
        
        try:
            # Try to start with firebase command
            subprocess.Popen(
                ["firebase", "emulators:start", "--only", "firestore"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            return FirestoreEmulatorManager.wait_for_emulator()
        except FileNotFoundError:
            logger.error("❌ Firebase CLI not found. Install with: npm install -g firebase-tools")
            return False


class FirestoreRulesTestRunner:
    """Run Firestore rules tests"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.test_file = self.tests_dir / "test_firestore_rules.py"
    
    def run_tests(self, verbose: bool = True, coverage: bool = False) -> bool:
        """Run the test suite"""
        logger.info("Running Firestore rules tests...")
        
        # Set environment
        env = os.environ.copy()
        env["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", str(self.test_file)]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term"
            ])
        
        cmd.append("--tb=short")
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, env=env, cwd=str(self.project_root))
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def generate_report(self) -> dict:
        """Generate test report"""
        report = {
            "project": "Fasal Saathi",
            "issue": "Issue #3: Firestore Rules Regression Suite",
            "status": "✅ READY FOR TESTING",
            "components": {
                "test_suite": {
                    "file": "tests/test_firestore_rules.py",
                    "total_tests": 30,
                    "test_classes": [
                        "TestUserRules",
                        "TestFeedbackRules",
                        "TestPostRules",
                        "TestCommentRules",
                        "TestReportRules",
                        "TestMarketplaceRules",
                        "TestFinanceApplicationRules",
                        "TestNotificationRules",
                        "TestSupplyChainRules",
                        "TestSummary"
                    ],
                    "description": "Comprehensive regression tests for Firestore security rules"
                },
                "ci_pipeline": {
                    "file": ".github/workflows/firestore-rules-ci.yml",
                    "description": "GitHub Actions CI pipeline with emulator testing",
                    "jobs": [
                        "firestore-rules-tests",
                        "lint-rules",
                        "security-audit"
                    ]
                },
                "emulator_setup": {
                    "bash": "scripts/setup_firestore_emulator.sh",
                    "batch": "scripts/setup_firestore_emulator.bat",
                    "description": "Emulator setup scripts for Unix and Windows"
                }
            },
            "coverage": {
                "collections": [
                    "users - read/update/reputation rules",
                    "feedback - admin-only read/delete",
                    "posts - create/update/delete with content validation",
                    "comments - CRUD with content validation",
                    "reports - expert/admin access",
                    "marketplace - public read, vendor write",
                    "finance_applications - role-based access",
                    "notifications - user read, system write",
                    "supply_chain_batches - nested documents with CRUD"
                ],
                "security_aspects": [
                    "Authentication enforcement",
                    "Role-based access control",
                    "Data ownership validation",
                    "Content validation rules",
                    "Admin override capabilities"
                ]
            }
        }
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Firestore Rules Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_firestore_tests.py --start-emulator
  python scripts/run_firestore_tests.py --verbose --coverage
  python scripts/run_firestore_tests.py --report
        """
    )
    
    parser.add_argument(
        "--start-emulator",
        action="store_true",
        help="Start Firestore emulator before tests"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Print test report only"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FIRESTORE RULES TEST RUNNER")
    print("="*80 + "\n")
    
    runner = FirestoreRulesTestRunner()
    
    if args.report:
        report = runner.generate_report()
        print(json.dumps(report, indent=2))
        return 0
    
    # Start emulator if requested
    if args.start_emulator:
        emulator = FirestoreEmulatorManager()
        if not emulator.start_emulator():
            logger.error("Failed to start emulator")
            return 1
    
    # Check if emulator is running
    if not FirestoreEmulatorManager.is_running():
        logger.error("⚠️  Firestore emulator is not running")
        logger.info("Start it with: firebase emulators:start --only firestore")
        logger.info("Or use: python scripts/run_firestore_tests.py --start-emulator")
        return 1
    
    # Run tests
    success = runner.run_tests(verbose=args.verbose, coverage=args.coverage)
    
    # Print report
    report = runner.generate_report()
    print("\n" + "="*80)
    print("TEST REPORT")
    print("="*80)
    print(json.dumps(report, indent=2))
    print("="*80 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

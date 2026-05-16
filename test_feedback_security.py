#!/usr/bin/env python3
"""
Feedback Security Test
Tests the security fixes for NoSQL injection prevention in the feedback system.
"""

import json
import requests
import sys
from datetime import datetime

# Test server URL (adjust as needed)
BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def test_valid_feedback():
    """Test valid feedback submission"""
    print_header("Test 1: Valid Feedback Submission")
    
    data = {
        "name": "John Farmer",
        "cropType": "Rice",
        "location": "Nashik, Maharashtra",
        "category": "feature",
        "message": "Great app! Please add more crop varieties.",
        "rating": 5,
        "userId": "test_user_123",
        "userEmail": "john@example.com"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/feedback", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Valid feedback accepted")
            return True
        else:
            print("❌ Valid feedback rejected")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_xss_injection():
    """Test XSS injection attempts"""
    print_header("Test 2: XSS Injection Attempts")
    
    test_cases = [
        {
            "name": "Test with script tag",
            "data": {
                "name": "<script>alert('xss')</script>",
                "message": "Test message",
                "rating": 3
            },
            "should_reject": True
        },
        {
            "name": "Test with event handler",
            "data": {
                "name": "Test User",
                "message": "Test onclick='alert(1)' message",
                "rating": 3
            },
            "should_reject": True
        },
        {
            "name": "Test with javascript protocol",
            "data": {
                "name": "Test User",
                "message": "Click here: javascript:alert(1)",
                "rating": 3
            },
            "should_reject": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print(f"Input: {json.dumps(test['data'], indent=2)}")
        
        try:
            response = requests.post(f"{BASE_URL}/api/feedback", json=test['data'])
            
            if test['should_reject']:
                if response.status_code == 400:
                    print(f"✅ Correctly rejected XSS attempt (Status: {response.status_code})")
                    passed += 1
                else:
                    print(f"❌ FAILED: XSS attempt was not rejected (Status: {response.status_code})")
            else:
                if response.status_code == 200:
                    print(f"✅ Correctly accepted safe input (Status: {response.status_code})")
                    passed += 1
                else:
                    print(f"❌ FAILED: Safe input was rejected (Status: {response.status_code})")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\nXSS Tests: {passed}/{total} passed")
    return passed == total

def test_nosql_injection():
    """Test NoSQL injection attempts"""
    print_header("Test 3: NoSQL Injection Attempts")
    
    test_cases = [
        {
            "name": "MongoDB operator injection",
            "data": {
                "name": "Test User",
                "message": "{$set: {admin: true}}",
                "rating": 1
            },
            "should_reject": True
        },
        {
            "name": "Nested object injection",
            "data": {
                "name": "Test User",
                "message": "Normal message",
                "rating": 1,
                "metadata": {"$ne": null}
            },
            "should_reject": True
        },
        {
            "name": "Path traversal attempt",
            "data": {
                "name": "Test User",
                "message": "../../etc/passwd",
                "rating": 1
            },
            "should_reject": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print(f"Input: {json.dumps(test['data'], indent=2)}")
        
        try:
            response = requests.post(f"{BASE_URL}/api/feedback", json=test['data'])
            
            if test['should_reject']:
                if response.status_code == 400:
                    print(f"✅ Correctly rejected NoSQL injection (Status: {response.status_code})")
                    passed += 1
                else:
                    print(f"❌ FAILED: NoSQL injection was not rejected (Status: {response.status_code})")
            else:
                if response.status_code == 200:
                    print(f"✅ Correctly accepted safe input (Status: {response.status_code})")
                    passed += 1
                else:
                    print(f"❌ FAILED: Safe input was rejected (Status: {response.status_code})")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\nNoSQL Injection Tests: {passed}/{total} passed")
    return passed == total

def test_input_validation():
    """Test input validation rules"""
    print_header("Test 4: Input Validation Rules")
    
    test_cases = [
        {
            "name": "Empty message",
            "data": {
                "name": "Test User",
                "message": "",
                "rating": 3
            },
            "should_reject": True
        },
        {
            "name": "Message too short",
            "data": {
                "name": "Test User",
                "message": "ab",
                "rating": 3
            },
            "should_reject": True
        },
        {
            "name": "Invalid rating (0)",
            "data": {
                "name": "Test User",
                "message": "Valid message",
                "rating": 0
            },
            "should_reject": True
        },
        {
            "name": "Invalid rating (6)",
            "data": {
                "name": "Test User",
                "message": "Valid message",
                "rating": 6
            },
            "should_reject": True
        },
        {
            "name": "Invalid category",
            "data": {
                "name": "Test User",
                "message": "Valid message",
                "rating": 3,
                "category": "invalid_category"
            },
            "should_reject": False  # Should default to "general"
        },
        {
            "name": "Invalid crop type",
            "data": {
                "name": "Test User",
                "message": "Valid message",
                "rating": 3,
                "cropType": "InvalidCrop"
            },
            "should_reject": False  # Should be ignored/set to None
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        try:
            response = requests.post(f"{BASE_URL}/api/feedback", json=test['data'])
            
            if test['should_reject']:
                if response.status_code == 400:
                    print(f"✅ Correctly rejected invalid input (Status: {response.status_code})")
                    passed += 1
                else:
                    print(f"❌ FAILED: Invalid input was accepted (Status: {response.status_code})")
            else:
                if response.status_code == 200:
                    print(f"✅ Correctly handled input (Status: {response.status_code})")
                    passed += 1
                elif response.status_code == 400:
                    # Check if it's for a different reason
                    result = response.json()
                    if "default" in result.get("message", "").lower():
                        print(f"✅ Correctly used default value (Status: {response.status_code})")
                        passed += 1
                    else:
                        print(f"❌ FAILED: Valid input was rejected (Status: {response.status_code})")
                else:
                    print(f"❌ Unexpected status: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\nInput Validation Tests: {passed}/{total} passed")
    return passed == total

def test_api_endpoints():
    """Test all API endpoints"""
    print_header("Test 5: API Endpoint Availability")
    
    endpoints = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/predict", "Prediction endpoint"),
        ("GET", "/api/feedback/stats", "Feedback stats"),
        ("GET", "/api/feedback/validate-test", "Validation test")
    ]
    
    passed = 0
    total = len(endpoints)
    
    for method, endpoint, description in endpoints:
        print(f"\nTesting: {description} ({method} {endpoint})")
        
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}")
            else:
                response = requests.post(f"{BASE_URL}{endpoint}")
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code < 500:  # Not a server error
                print(f"✅ Endpoint available")
                passed += 1
            else:
                print(f"❌ Server error")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\nAPI Endpoint Tests: {passed}/{total} passed")
    return passed == total

def main():
    """Run all security tests"""
    print("="*60)
    print("FEEDBACK SYSTEM SECURITY TEST SUITE")
    print("="*60)
    print(f"Testing server: {BASE_URL}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("\n❌ Server is not responding correctly")
            print(f"Status: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Please make sure the FastAPI server is running on port 8000")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Valid Feedback", test_valid_feedback),
        ("XSS Injection", test_xss_injection),
        ("NoSQL Injection", test_nosql_injection),
        ("Input Validation", test_input_validation),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    # Security assessment
    print("\n" + "="*60)
    print("SECURITY ASSESSMENT")
    print("="*60)
    
    if passed == total:
        print("✅ EXCELLENT - All security tests passed!")
        print("The feedback system is protected against:")
        print("  - NoSQL injection attacks")
        print("  - XSS (Cross-Site Scripting) attacks")
        print("  - Input validation bypass attempts")
        print("  - Path traversal attempts")
    elif passed >= total * 0.8:
        print("⚠️  GOOD - Most security tests passed")
        print("The feedback system has good security but may need improvements.")
    elif passed >= total * 0.6:
        print("⚠️  FAIR - Some security tests passed")
        print("The feedback system has basic security but needs significant improvements.")
    else:
        print("❌ POOR - Most security tests failed")
        print("The feedback system has critical security vulnerabilities!")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code based on results
    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
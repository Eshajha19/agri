import os
import pytest
from sustainability_analytics import SustainabilityAnalytics

def test_sustainability_persistence_across_restarts():
    # 1. Instantiate the first engine and generate records
    engine1 = SustainabilityAnalytics()
    # Ensure test isolation by overriding is_testing to write to a specific test file
    engine1.is_testing = True
    test_file = engine1._get_local_file_path()
    if os.path.exists(test_file):
        os.remove(test_file)

    payload = {
        "crop_type": "Rice",
        "season": "Kharif",
        "acreage": 2.5,
        "irrigation_type": "drip",
        "irrigation_events": 10,
        "organic_practices": True,
        "user_id": "persisted-farmer-123"
    }
    
    # Analyze and save
    res = engine1.analyze(payload)
    record_id = res["record_id"]

    # Verify record was appended in the first engine instance
    hist1 = engine1.get_history("persisted-farmer-123")
    assert len(hist1) == 1
    assert hist1[0]["record_id"] == record_id
    assert hist1[0]["crop_type"] == "Rice"
    
    # 2. Simulate server restart by creating a completely new engine instance
    engine2 = SustainabilityAnalytics()
    # Override is_testing to point to the same test file to simulate persistence retrieval
    engine2.is_testing = True
    
    # Retrieve history from the second instance
    hist2 = engine2.get_history("persisted-farmer-123")
    
    # 3. Assert that the history was successfully loaded/recovered from durable storage
    assert len(hist2) == 1
    assert hist2[0]["record_id"] == record_id
    assert hist2[0]["crop_type"] == "Rice"
    assert hist2[0]["season"] == "Kharif"
    assert hist2[0]["acreage"] == 2.5

    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)

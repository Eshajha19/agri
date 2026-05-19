import time
import pytest
from main import IdempotencyCache

def test_idempotency_cache_bounded_capacity():
    # Initialize cache with small capacity
    cache = IdempotencyCache(max_size=3, ttl_seconds=60)
    
    cache["key1"] = "val1"
    cache["key2"] = "val2"
    cache["key3"] = "val3"
    
    assert len(cache) == 3
    assert "key1" in cache
    
    # Adding a 4th key should evict the oldest key (key1)
    cache["key4"] = "val4"
    assert len(cache) == 3
    assert "key1" not in cache
    assert "key2" in cache
    assert "key3" in cache
    assert "key4" in cache

def test_idempotency_cache_ttl_expiration():
    # Initialize cache with 1 second TTL
    cache = IdempotencyCache(max_size=5, ttl_seconds=1)
    
    cache["key1"] = "val1"
    assert "key1" in cache
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    assert "key1" not in cache
    assert len(cache) == 0

def test_idempotency_cache_clear():
    cache = IdempotencyCache(max_size=5, ttl_seconds=60)
    cache["key1"] = "val1"
    cache["key2"] = "val2"
    
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0

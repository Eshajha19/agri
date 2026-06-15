from backend.main import app

def test_no_duplicate_routes():
    seen = set()
    for route in app.routes:
        key = (route.path, tuple(route.methods))
        assert key not in seen, f"Duplicate route detected: {key}"
        seen.add(key)

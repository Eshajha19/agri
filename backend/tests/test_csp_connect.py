def test_csp_allows_backend_connect(client):
    resp = client.get("/health")
    assert "connect-src" in resp.headers.get("Content-Security-Policy", "")
    assert "your-backend.onrender.com" in resp.headers["Content-Security-Policy"]

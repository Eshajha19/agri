def test_missing_static_asset_returns_fallback(client):
    resp = client.get("/static/missing.png")
    assert resp.status_code == 404
    assert "Static asset not found" in resp.text

def test_cross_tenant_access_denied(client):
    resp = client.get("/tenants/tenantA/reports", headers={"Authorization": "Bearer token-for-tenantB"})
    assert resp.status_code == 403

def test_same_tenant_access_allowed(client):
    resp = client.get("/tenants/tenantA/reports", headers={"Authorization": "Bearer token-for-tenantA"})
    assert resp.status_code == 200

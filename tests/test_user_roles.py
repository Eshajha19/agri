def test_user_roles_authenticated(client, firebase_token):
    response = client.get("/user_roles", headers={"Authorization": f"Bearer {firebase_token}"})
    assert response.status_code == 200
    data = response.json()
    assert "roles" in data
    assert isinstance(data["roles"], list)

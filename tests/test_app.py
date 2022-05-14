from fastapi.testclient import TestClient
from api.app import app

client: TestClient = TestClient(app)


def test_get_api():
    """ Test connection to API """
    response = client.get(url="/api/")
    assert response.status_code == 200

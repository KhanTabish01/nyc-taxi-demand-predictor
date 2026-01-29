import requests


def test_health_endpoint():
    """Basic health check for the API."""
    resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert data.get("model_loaded") is True

"""Pytest-based API endpoint tests (requires running API)."""
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert data.get("model_loaded") is True


def test_info_endpoint():
    resp = requests.get(f"{BASE_URL}/info", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("model_type") is not None
    assert len(data.get("features", [])) > 0


def test_predict_endpoint():
    prediction_input = {
        "hour_sin": 0.9511,
        "hour_cos": 0.309,
        "dow_sin": 0.0,
        "dow_cos": 1.0,
        "month_sin": 0.2588,
        "month_cos": 0.9659,
        "lag_1h": 45.5,
        "lag_24h": 38.2,
        "lag_168h": 40.1,
        "diff_24h": 7.3,
        "rolling_7d_mean": 42.0,
        "rolling_7d_std": 8.5,
        "rolling_14d_mean": 41.5,
        "rolling_7d_cv": 0.2,
        "zone_mean_demand": 35.0,
        "zone_rank": 10,
        "zone_is_top50": 1,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=prediction_input, timeout=5)
    assert resp.status_code == 200
    result = resp.json()
    assert result.get("predicted_pickups") is not None
    assert result.get("predicted_pickups") >= 0

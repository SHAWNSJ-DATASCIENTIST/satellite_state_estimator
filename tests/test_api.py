"""
Integration tests for Flask endpoints.
"""
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_dashboard(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"ORBITAL-EKF // STATE ESTIMATOR" in rv.data

def test_telemetry_post(client):
    payload = {
        "rtc_time": 1728245100.0,
        "bmp280": {"pressure_pa": 0.02, "temperature_c": -15.0},
        "mpu6050": {
            "accel_x": 0.01, "accel_y": 0.02, "accel_z": 0.05,
            "gyro_x": 0.01, "gyro_y": 0.01, "gyro_z": 0.02
        },
        "lsm6dsm": {"mag_x": 21.0, "mag_y": -4.0, "mag_z": 42.0}
    }
    rv = client.post('/api/telemetry', json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "attitude" in data
    assert "next_heading" in data
    assert "current_heading_deg" in data["next_heading"]
    assert "next_heading_deg" in data["next_heading"]
    assert "orbit_prediction" in data

def test_telemetry_invalid_json(client):
    rv = client.post('/api/telemetry', data="not-a-json", content_type="application/json")
    assert rv.status_code == 400

def test_simulation_step(client):
    rv = client.post('/api/simulation/step', json={"scenario": "stable_orbit"})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert data["orbit_prediction"]["orbit_status"] in ["STABLE_ORBIT", "ORBITAL_DECAY"]

def test_reset(client):
    rv = client.post('/api/reset')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True

"""
Unit tests for Attitude EKF and Altitude KF.
"""
import math
import numpy as np
import pytest
from satellite.kalman.attitude_ekf import AttitudeEKF
from satellite.kalman.altitude_kf import AltitudeKalmanFilter
from satellite.kalman.fused_estimator import FusedSatelliteEstimator
from satellite.sensors import SensorParser

def test_attitude_ekf_initialization():
    ekf = AttitudeEKF()
    assert len(ekf.q) == 4
    assert np.isclose(np.linalg.norm(ekf.q), 1.0)
    assert np.allclose(ekf.gyro_bias, 0.0)

def test_attitude_ekf_heading_prediction():
    ekf = AttitudeEKF()
    
    # Predict with steady z-axis spin (yaw rate) = 0.1 rad/s (~5.73 deg/s)
    gyro = np.array([0.0, 0.0, 0.1])
    dt = 0.5
    ekf.predict(gyro, dt)
    
    res = ekf.predict_next_heading(lookahead_sec=1.0)
    assert "current_heading_deg" in res
    assert "next_heading_deg" in res
    assert "heading_rate_deg_s" in res
    assert 0.0 <= res["current_heading_deg"] <= 360.0
    assert 0.0 <= res["next_heading_deg"] <= 360.0

def test_altitude_kf_filtering():
    alt_kf = AltitudeKalmanFilter(initial_altitude_m=300000.0)
    dt = 0.1
    
    # Simulate a satellite decaying at -10 m/s
    for i in range(10):
        alt_kf.predict(dt)
        # Raw noisy reading
        noisy_alt = 300000.0 - (10.0 * (i + 1) * dt) + np.random.normal(0, 0.5)
        alt_kf.update(noisy_alt, -0.01)
        
    assert alt_kf.altitude < 300000.0
    assert alt_kf.vertical_velocity < 0.0  # Demonstrates decay velocity is detected

def test_fused_estimator_pipeline():
    estimator = FusedSatelliteEstimator()
    parser = SensorParser()
    
    dummy_payload = {
        "rtc_time": 1728240000.0,
        "bmp280": {
            "pressure_pa": 0.005,
            "temperature_c": -30.0
        },
        "mpu6050": {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.05,
            "gyro_x": 0.01,
            "gyro_y": 0.0,
            "gyro_z": 0.02
        },
        "lsm6dsm": {
            "mag_x": 22.0,
            "mag_y": 0.0,
            "mag_z": 42.0
        }
    }
    
    packet = parser.parse_payload(dummy_payload)
    state = estimator.process_packet(packet)
    
    assert "attitude" in state
    assert "altitude_state" in state
    assert "current_heading_deg" in state["attitude"]
    assert "next_heading_deg" in state["attitude"]

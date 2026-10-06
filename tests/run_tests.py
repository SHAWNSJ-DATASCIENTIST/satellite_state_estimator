import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from satellite.kalman.attitude_ekf import AttitudeEKF
from satellite.kalman.altitude_kf import AltitudeKalmanFilter
from satellite.kalman.fused_estimator import FusedSatelliteEstimator
from satellite.sensors import SensorParser
from satellite.orbit_predictor import OrbitPredictor, OrbitStatus
from app import app

class TestKalmanFilters(unittest.TestCase):
    def test_attitude_ekf_initialization(self):
        ekf = AttitudeEKF()
        self.assertEqual(len(ekf.q), 4)
        self.assertTrue(np.isclose(np.linalg.norm(ekf.q), 1.0))
        self.assertTrue(np.allclose(ekf.gyro_bias, 0.0))

    def test_attitude_ekf_heading_prediction(self):
        ekf = AttitudeEKF()
        gyro = np.array([0.0, 0.0, 0.1])
        dt = 0.5
        ekf.predict(gyro, dt)
        res = ekf.predict_next_heading(lookahead_sec=1.0)
        self.assertIn("current_heading_deg", res)
        self.assertIn("next_heading_deg", res)
        self.assertIn("heading_rate_deg_s", res)
        self.assertTrue(0.0 <= res["current_heading_deg"] <= 360.0)
        self.assertTrue(0.0 <= res["next_heading_deg"] <= 360.0)

    def test_altitude_kf_filtering(self):
        alt_kf = AltitudeKalmanFilter(initial_altitude_m=300000.0)
        dt = 0.1
        for i in range(10):
            alt_kf.predict(dt)
            noisy_alt = 300000.0 - (10.0 * (i + 1) * dt)
            alt_kf.update(noisy_alt, -0.01)
        self.assertLess(alt_kf.altitude, 300000.0)
        self.assertLess(alt_kf.vertical_velocity, 0.0)

    def test_fused_estimator_pipeline(self):
        estimator = FusedSatelliteEstimator()
        parser = SensorParser()
        dummy_payload = {
            "rtc_time": 1728240000.0,
            "bmp280": {"pressure_pa": 0.005, "temperature_c": -30.0},
            "mpu6050": {
                "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.05,
                "gyro_x": 0.01, "gyro_y": 0.0, "gyro_z": 0.02
            },
            "lsm6dsm": {"mag_x": 22.0, "mag_y": 0.0, "mag_z": 42.0}
        }
        packet = parser.parse_payload(dummy_payload)
        state = estimator.process_packet(packet)
        self.assertIn("attitude", state)
        self.assertIn("altitude_state", state)
        self.assertIn("current_heading_deg", state["attitude"])
        self.assertIn("next_heading_deg", state["attitude"])

class TestOrbitPredictor(unittest.TestCase):
    def test_stable_orbit_evaluation(self):
        predictor = OrbitPredictor(mission_mode="ORBITAL")
        state = {
            "altitude_state": {
                "filtered_altitude_m": 350000.0,
                "vertical_velocity_mps": -0.02
            },
            "raw_sensors": {"bmp280": {"pressure_pa": 0.001}},
            "attitude": {
                "current_heading_deg": 45.0,
                "next_heading_deg": 45.5,
                "heading_rate_deg_s": 0.5,
                "tumble_rate_deg_s": 0.8
            }
        }
        res = predictor.evaluate_orbit(state)
        self.assertTrue(res["is_in_orbit"])
        self.assertEqual(res["orbit_status"], OrbitStatus.STABLE_ORBIT)
        self.assertGreater(res["stability_score"], 80.0)

    def test_critical_reentry_evaluation(self):
        predictor = OrbitPredictor(mission_mode="ORBITAL")
        state = {
            "altitude_state": {
                "filtered_altitude_m": 85000.0,
                "vertical_velocity_mps": -45.0
            },
            "raw_sensors": {"bmp280": {"pressure_pa": 25.0}},
            "attitude": {
                "current_heading_deg": 120.0,
                "next_heading_deg": 185.0,
                "heading_rate_deg_s": 65.0,
                "tumble_rate_deg_s": 75.0
            }
        }
        res = predictor.evaluate_orbit(state)
        self.assertFalse(res["is_in_orbit"])
        self.assertEqual(res["orbit_status"], OrbitStatus.CRITICAL_REENTRY)
        self.assertTrue(res["is_tumbling"])

class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_dashboard_route(self):
        rv = self.client.get('/')
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"ORBITAL-EKF // STATE ESTIMATOR", rv.data)

    def test_telemetry_post(self):
        payload = {
            "rtc_time": 1728245100.0,
            "bmp280": {"pressure_pa": 0.02, "temperature_c": -15.0},
            "mpu6050": {
                "accel_x": 0.01, "accel_y": 0.02, "accel_z": 0.05,
                "gyro_x": 0.01, "gyro_y": 0.01, "gyro_z": 0.02
            },
            "lsm6dsm": {"mag_x": 21.0, "mag_y": -4.0, "mag_z": 42.0}
        }
        rv = self.client.post('/api/telemetry', json=payload)
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertTrue(data["success"])
        self.assertIn("attitude", data)
        self.assertIn("next_heading", data)
        self.assertIn("current_heading_deg", data["next_heading"])
        self.assertIn("next_heading_deg", data["next_heading"])
        self.assertIn("orbit_prediction", data)

    def test_simulation_step(self):
        rv = self.client.post('/api/simulation/step', json={"scenario": "stable_orbit"})
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertTrue(data["success"])

if __name__ == '__main__':
    unittest.main(verbosity=2)

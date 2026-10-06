"""
Unit tests for Orbit Predictor logic.
"""
from satellite.orbit_predictor import OrbitPredictor, OrbitStatus

def test_stable_orbit_evaluation():
    predictor = OrbitPredictor(mission_mode="ORBITAL")
    state = {
        "altitude_state": {
            "filtered_altitude_m": 350000.0,
            "vertical_velocity_mps": -0.02
        },
        "raw_sensors": {
            "bmp280": {"pressure_pa": 0.001}
        },
        "attitude": {
            "current_heading_deg": 45.0,
            "next_heading_deg": 45.5,
            "heading_rate_deg_s": 0.5,
            "tumble_rate_deg_s": 0.8
        }
    }
    eval_res = predictor.evaluate_orbit(state)
    assert eval_res["is_in_orbit"] is True
    assert eval_res["orbit_status"] == OrbitStatus.STABLE_ORBIT
    assert eval_res["stability_score"] > 80.0

def test_critical_reentry_evaluation():
    predictor = OrbitPredictor(mission_mode="ORBITAL")
    state = {
        "altitude_state": {
            "filtered_altitude_m": 85000.0,  # Below Karman line (100 km)
            "vertical_velocity_mps": -45.0   # Plunging
        },
        "raw_sensors": {
            "bmp280": {"pressure_pa": 25.0}
        },
        "attitude": {
            "current_heading_deg": 120.0,
            "next_heading_deg": 185.0,
            "heading_rate_deg_s": 65.0,
            "tumble_rate_deg_s": 75.0        # Violently tumbling
        }
    }
    eval_res = predictor.evaluate_orbit(state)
    assert eval_res["is_in_orbit"] is False
    assert eval_res["orbit_status"] == OrbitStatus.CRITICAL_REENTRY
    assert eval_res["is_tumbling"] is True

def test_orbital_decay_evaluation():
    predictor = OrbitPredictor(mission_mode="ORBITAL")
    state = {
        "altitude_state": {
            "filtered_altitude_m": 140000.0,  # VLEO
            "vertical_velocity_mps": -2.5
        },
        "raw_sensors": {
            "bmp280": {"pressure_pa": 0.2}
        },
        "attitude": {
            "current_heading_deg": 10.0,
            "next_heading_deg": 11.2,
            "heading_rate_deg_s": 1.2,
            "tumble_rate_deg_s": 2.0
        }
    }
    eval_res = predictor.evaluate_orbit(state)
    assert eval_res["is_in_orbit"] is True
    assert eval_res["orbit_status"] == OrbitStatus.ORBITAL_DECAY
    assert eval_res["estimated_reentry_time_sec"] is not None

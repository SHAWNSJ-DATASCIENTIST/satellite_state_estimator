"""
Satellite State Estimation & Kalman Filter Flask Application.
Receives sensor telemetry (BMP280, MPU6050, LSM6DSM, RTC), executes Extended Kalman
Filtering for 3D attitude & altitude, computes the next heading, and predicts whether
the satellite is still in orbit.
"""
import threading
import time
from collections import deque
from flask import Flask, request, jsonify, render_template
from config import Config
from satellite.sensors import SensorParser
from satellite.kalman.fused_estimator import FusedSatelliteEstimator
from satellite.orbit_predictor import OrbitPredictor
from satellite.simulator import SatelliteFlightSimulator

app = Flask(__name__)
app.config.from_object(Config)

# Global State & Pipeline
history_lock = threading.Lock()
telemetry_history = deque(maxlen=Config.MAX_TELEMETRY_HISTORY)

parser = SensorParser()
estimator = FusedSatelliteEstimator(mission_mode=Config.MISSION_MODE)
predictor = OrbitPredictor(mission_mode=Config.MISSION_MODE)
simulator = SatelliteFlightSimulator(scenario="stable_orbit")

# Background simulation runner
sim_thread = None
sim_running = False
sim_lock = threading.Lock()
latest_evaluation = None


def process_telemetry_dict(data: dict) -> dict:
    """Processes a raw telemetry packet through the Kalman filter & orbit predictor pipeline."""
    global latest_evaluation
    
    # 1. Parse and normalize sensors
    packet = parser.parse_payload(data)
    
    # 2. Run Fused Kalman Filter
    state = estimator.process_packet(packet, lookahead_sec=Config.PREDICTION_LOOKAHEAD_SEC)
    
    # 3. Predict Orbit Decay & Health
    orbit_eval = predictor.evaluate_orbit(state)
    
    # Combined response
    response_payload = {
        "success": True,
        "timestamp": state["timestamp"],
        "step": state["step"],
        "dt": state["dt"],
        "attitude": state["attitude"],
        "altitude_state": state["altitude_state"],
        "orbit_prediction": orbit_eval,
        "next_heading": {
            "current_heading_deg": state["attitude"]["current_heading_deg"],
            "next_heading_deg": state["attitude"]["next_heading_deg"],
            "heading_rate_deg_s": state["attitude"]["heading_rate_deg_s"],
            "trajectory": state["attitude"]["trajectory"]
        },
        "raw_sensors": state["raw_sensors"]
    }
    
    with history_lock:
        telemetry_history.append(response_payload)
        latest_evaluation = response_payload

    return response_payload


def background_simulation_loop():
    """Background worker for streaming simulation data."""
    global sim_running
    while sim_running:
        try:
            raw_pkt = simulator.step()
            process_telemetry_dict(raw_pkt)
            time.sleep(0.1)  # 10 Hz
        except Exception as e:
            print(f"Simulation loop error: {e}")
            break


@app.route('/')
def index():
    """Mission Control Web Dashboard."""
    return render_template('index.html', mode=Config.MISSION_MODE)


@app.route('/api/telemetry', methods=['POST'])
def ingest_telemetry():
    """
    Ingest single sensor packet from external satellite hardware or script.
    Expected JSON keys: bmp280, mpu6050, lsm6dsm, rtc_time
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or missing JSON payload"}), 400

    try:
        result = process_telemetry_dict(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/telemetry/batch', methods=['POST'])
def ingest_batch_telemetry():
    """Ingest array of telemetry packets for sequential replay / batch analysis."""
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, list):
        return jsonify({"success": False, "error": "Expected a JSON list of telemetry packets"}), 400

    results = []
    try:
        for pkt in data:
            res = process_telemetry_dict(pkt)
            results.append(res)
        return jsonify({
            "success": True,
            "processed_count": len(results),
            "final_state": results[-1] if results else None
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "processed_count": len(results)}), 400


@app.route('/api/state', methods=['GET'])
def get_current_state():
    """Returns the latest estimated state and orbit prediction."""
    with history_lock:
        if latest_evaluation is not None:
            return jsonify({"success": True, "state": latest_evaluation})
        else:
            return jsonify({
                "success": False,
                "message": "No telemetry processed yet. Send telemetry to /api/telemetry or start simulation."
            }), 404


@app.route('/api/history', methods=['GET'])
def get_telemetry_history():
    """Returns recent telemetry history buffer for charts."""
    limit = request.args.get('limit', default=100, type=int)
    with history_lock:
        hist_slice = list(telemetry_history)[-limit:]
    return jsonify({
        "success": True,
        "count": len(hist_slice),
        "history": hist_slice
    })


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Resets Kalman filter states and clears telemetry history."""
    global latest_evaluation
    init_alt = request.json.get('initial_altitude_m', 300000.0) if request.json else 300000.0
    
    with history_lock:
        telemetry_history.clear()
        latest_evaluation = None
        parser.last_timestamp = None
        estimator.reset(initial_altitude_m=init_alt)
        
    return jsonify({"success": True, "message": "Kalman filters and telemetry buffer reset successfully."})


@app.route('/api/simulation/scenarios', methods=['GET'])
def get_simulation_scenarios():
    """Lists available simulated flight scenarios."""
    scenarios = [
        {
            "id": "stable_orbit",
            "name": "Stable LEO Orbit",
            "description": "Satellite at ~350 km altitude, vacuum pressure, stable spin attitude, nominal heading.",
            "expected_status": "STABLE_ORBIT"
        },
        {
            "id": "orbital_decay",
            "name": "Orbital Decay (VLEO)",
            "description": "Satellite at ~135 km with atmospheric drag, continuous negative vertical rate, attitude drift.",
            "expected_status": "ORBITAL_DECAY"
        },
        {
            "id": "critical_reentry",
            "name": "Atmospheric Re-entry & Tumble",
            "description": "Satellite plummeting below 85 km Kármán boundary, rapid pressure surge, violent multi-axis tumbling.",
            "expected_status": "CRITICAL_REENTRY"
        },
        {
            "id": "suborbital_cansat",
            "name": "Model Satellite / CanSat Flight",
            "description": "Model rocket suborbital apogee at 2.5 km and parachute descent towards ground landing.",
            "expected_status": "SUBORBITAL_TEST"
        }
    ]
    return jsonify({"success": True, "scenarios": scenarios})


@app.route('/api/simulation/step', methods=['POST'])
def step_simulation():
    """Generates and processes one single synthetic step."""
    scenario = request.json.get('scenario') if request.json else None
    if scenario and scenario != simulator.scenario:
        simulator.reset(scenario)

    raw_pkt = simulator.step()
    result = process_telemetry_dict(raw_pkt)
    return jsonify(result)


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Starts background real-time simulation stream (10 Hz)."""
    global sim_thread, sim_running
    scenario = request.json.get('scenario', 'stable_orbit') if request.json else 'stable_orbit'
    
    with sim_lock:
        simulator.reset(scenario)
        if not sim_running:
            sim_running = True
            sim_thread = threading.Thread(target=background_simulation_loop, daemon=True)
            sim_thread.start()

    return jsonify({
        "success": True,
        "message": f"Real-time simulation started with scenario: '{scenario}'",
        "scenario": scenario
    })


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stops the real-time background simulation."""
    global sim_running
    with sim_lock:
        sim_running = False
    return jsonify({"success": True, "message": "Simulation stopped."})


@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Checks whether the background simulation is running."""
    return jsonify({
        "success": True,
        "is_running": sim_running,
        "active_scenario": simulator.scenario
    })


@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Read or modify runtime parameters."""
    if request.method == 'POST':
        cfg_data = request.get_json(force=True, silent=True) or {}
        if 'mission_mode' in cfg_data:
            Config.MISSION_MODE = cfg_data['mission_mode']
            estimator.mission_mode = Config.MISSION_MODE
            predictor.mission_mode = Config.MISSION_MODE
        if 'prediction_lookahead_sec' in cfg_data:
            Config.PREDICTION_LOOKAHEAD_SEC = float(cfg_data['prediction_lookahead_sec'])
        if 'mag_declination_deg' in cfg_data:
            Config.MAG_DECLINATION_DEG = float(cfg_data['mag_declination_deg'])
            estimator.attitude_ekf.mag_declination_rad = (Config.MAG_DECLINATION_DEG * 3.14159265 / 180.0)
            
        return jsonify({"success": True, "message": "Configuration updated successfully."})
        
    return jsonify({
        "success": True,
        "mission_mode": Config.MISSION_MODE,
        "prediction_lookahead_sec": Config.PREDICTION_LOOKAHEAD_SEC,
        "mag_declination_deg": Config.MAG_DECLINATION_DEG,
        "karman_line_altitude_m": Config.KARMAN_LINE_ALTITUDE_M,
        "leo_min_stable_altitude_m": Config.LEO_MIN_STABLE_ALTITUDE_M
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "service": "satellite-kalman-estimator",
        "version": "1.0.0"
    })


if __name__ == '__main__':
    print("=================================================================")
    print(" Orbital Kalman Filter & State Estimator Dashboard Launching")
    print(" Access Web Mission Control at: http://127.0.0.1:5000")
    print(" Telemetry ingestion endpoint: POST http://127.0.0.1:5000/api/telemetry")
    print("=================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)

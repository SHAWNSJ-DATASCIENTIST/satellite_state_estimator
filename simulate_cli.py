"""
Command-line Satellite Telemetry Streamer & Testing Tool.
Streams simulated BMP280, MPU6050, LSM6DSM, and RTC data to the Flask REST API,
or runs local offline estimation to demonstrate Kalman filtering and orbit prediction.
"""
import argparse
import json
import time
import requests
from satellite.simulator import SatelliteFlightSimulator
from satellite.sensors import SensorParser
from satellite.kalman.fused_estimator import FusedSatelliteEstimator
from satellite.orbit_predictor import OrbitPredictor

def run_local_demo(scenario: str, steps: int = 50, dt: float = 0.1):
    """Executes state estimation locally in terminal without requiring HTTP requests."""
    print("=" * 70)
    print(f" SATELLITE KALMAN FILTER & STATE ESTIMATION CLI DEMO")
    print(f" Scenario: {scenario.upper()} | Steps: {steps} | Interval: {dt}s")
    print("=" * 70)

    sim = SatelliteFlightSimulator(scenario=scenario)
    parser = SensorParser()
    estimator = FusedSatelliteEstimator()
    predictor = OrbitPredictor()

    for i in range(1, steps + 1):
        raw_pkt = sim.step()
        packet = parser.parse_payload(raw_pkt)
        state = estimator.process_packet(packet)
        orbit = predictor.evaluate_orbit(state)

        att = state["attitude"]
        alt = state["altitude_state"]

        print(
            f"Step {i:03d} | "
            f"Alt: {alt['filtered_altitude_m']/1000.0:6.1f} km | "
            f"Vz: {alt['vertical_velocity_mps']:+6.2f} m/s | "
            f"Head: {att['current_heading_deg']:5.1f} deg -> Next: {att['next_heading_deg']:5.1f} deg | "
            f"Tumble: {att['tumble_rate_deg_s']:5.1f} deg/s | "
            f"Status: {orbit['orbit_status']:15s} | "
            f"In-Orbit: {'YES' if orbit['is_in_orbit'] else 'NO '}"
        )
        time.sleep(dt)

    print("=" * 70)
    print(" Simulation completed.")

def stream_to_api(api_url: str, scenario: str, steps: int = 100, dt: float = 0.1):
    """Streams synthetic telemetry packets over HTTP to a running Flask server."""
    endpoint = f"{api_url.rstrip('/')}/api/telemetry"
    print(f"Streaming {scenario} telemetry to {endpoint} ...")
    sim = SatelliteFlightSimulator(scenario=scenario)

    for i in range(1, steps + 1):
        raw_pkt = sim.step()
        try:
            res = requests.post(endpoint, json=raw_pkt, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                att = data["attitude"]
                alt = data["altitude_state"]
                orbit = data["orbit_prediction"]
                print(
                    f"[{i:03d}] Alt: {alt['filtered_altitude_m']/1000.0:.1f}km | "
                    f"Heading: {att['current_heading_deg']:.1f}° -> {att['next_heading_deg']:.1f}° | "
                    f"Status: {orbit['orbit_status']}"
                )
            else:
                print(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Connection failed: {e}")
            break
        time.sleep(dt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Satellite Kalman Telemetry Streamer")
    parser.add_argument('--mode', choices=['local', 'stream'], default='local', help='Execution mode')
    parser.add_argument('--scenario', choices=['stable_orbit', 'orbital_decay', 'critical_reentry', 'suborbital_cansat'],
                        default='stable_orbit', help='Flight scenario')
    parser.add_argument('--steps', type=int, default=30, help='Number of telemetry steps')
    parser.add_argument('--interval', type=float, default=0.08, help='Delay between steps (sec)')
    parser.add_argument('--url', default='http://127.0.0.1:5000', help='Flask API URL (for stream mode)')

    args = parser.parse_args()
    if args.mode == 'local':
        run_local_demo(args.scenario, args.steps, args.interval)
    else:
        stream_to_api(args.url, args.scenario, args.steps, args.interval)

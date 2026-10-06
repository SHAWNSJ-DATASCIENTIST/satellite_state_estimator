"""
Orbit Determination and De-orbit Prediction Engine.
Evaluates estimated altitude, decay rate (dh/dt), pressure spikes, and next heading stability
to predict whether the satellite is still in orbit or undergoing orbital decay/re-entry.
"""
import math
from config import Config

class OrbitStatus:
    STABLE_ORBIT = "STABLE_ORBIT"
    ORBITAL_DECAY = "ORBITAL_DECAY"
    CRITICAL_REENTRY = "CRITICAL_REENTRY"
    DEORBITED = "DEORBITED"

class OrbitPredictor:
    """
    Predicts whether a model/space satellite is still in orbit using:
    - Kalman filtered altitude and descent rate (dh/dt)
    - BMP280 barometric pressure (vacuum vs atmospheric reentry)
    - Gyro/Mag attitude stability and predicted next heading
    """
    def __init__(self, mission_mode: str = Config.MISSION_MODE):
        self.mission_mode = mission_mode

    def evaluate_orbit(self, state: dict) -> dict:
        """
        Takes state dictionary from FusedSatelliteEstimator and produces
        orbital status and predictive health metrics.
        """
        alt = state["altitude_state"]["filtered_altitude_m"]
        vz = state["altitude_state"]["vertical_velocity_mps"]
        pressure_pa = state["raw_sensors"]["bmp280"]["pressure_pa"]
        
        att = state["attitude"]
        curr_heading = att["current_heading_deg"]
        next_heading = att["next_heading_deg"]
        heading_rate = att["heading_rate_deg_s"]
        tumble_rate = att["tumble_rate_deg_s"]
        
        # Decide boundary based on mission mode
        if self.mission_mode == "SUBORBITAL_TEST":
            min_boundary = Config.SUBORBITAL_FLOOR_M
            nominal_target = Config.SUBORBITAL_CEILING_M
        else:
            min_boundary = Config.KARMAN_LINE_ALTITUDE_M
            nominal_target = Config.LEO_MIN_STABLE_ALTITUDE_M

        # 1. Attitude Tumbling Check
        is_tumbling = tumble_rate > Config.TUMBLE_THRESHOLD_DEG_S
        attitude_unstable = (tumble_rate > Config.MODERATE_TUMBLE_DEG_S) or (abs(heading_rate) > 20.0)

        # 2. Orbital Status Classification
        is_in_orbit = True
        status = OrbitStatus.STABLE_ORBIT
        reason = []
        severity = "nominal"  # nominal, warning, critical, failure

        if self.mission_mode == "ORBITAL":
            # Space satellite criteria
            if alt < 30000.0 or pressure_pa > 2000.0:
                is_in_orbit = False
                status = OrbitStatus.DEORBITED
                severity = "failure"
                reason.append(f"Altitude ({alt/1000.0:.1f} km) is in dense troposphere/stratosphere; atmospheric flight.")
            elif alt < Config.KARMAN_LINE_ALTITUDE_M:
                # Below Karman line (100 km)
                is_in_orbit = False
                status = OrbitStatus.CRITICAL_REENTRY
                severity = "critical"
                reason.append(f"Satellite has breached the Kármán line (100 km). Current altitude: {alt/1000.0:.1f} km.")
            elif alt < Config.LEO_MIN_STABLE_ALTITUDE_M:
                # 100 km - 160 km: Decaying orbit
                if vz < Config.CRITICAL_DESCENT_RATE_MPS or is_tumbling:
                    status = OrbitStatus.CRITICAL_REENTRY
                    severity = "critical"
                    reason.append(f"Very Low Earth Orbit ({alt/1000.0:.1f} km) with high descent rate ({vz:.1f} m/s) and tumbling ({tumble_rate:.1f}°/s).")
                else:
                    status = OrbitStatus.ORBITAL_DECAY
                    severity = "warning"
                    reason.append(f"Sub-nominal altitude ({alt/1000.0:.1f} km). Noticeable atmospheric drag causing decay.")
            else:
                # Above 160 km
                if vz < Config.CRITICAL_DESCENT_RATE_MPS:
                    status = OrbitStatus.ORBITAL_DECAY
                    severity = "warning"
                    reason.append(f"High altitude ({alt/1000.0:.1f} km) but experiencing rapid descent ({vz:.1f} m/s).")
                elif is_tumbling:
                    status = OrbitStatus.STABLE_ORBIT
                    severity = "warning"
                    reason.append(f"In orbit ({alt/1000.0:.1f} km) but attitude control lost: tumbling at {tumble_rate:.1f}°/s.")
                else:
                    status = OrbitStatus.STABLE_ORBIT
                    severity = "nominal"
                    reason.append(f"Nominal stable orbit at {alt/1000.0:.1f} km altitude. Stable heading drift.")
        else:
            # Model satellite / Suborbital CanSat mode
            if alt <= min_boundary:
                is_in_orbit = False
                status = OrbitStatus.DEORBITED
                severity = "failure"
                reason.append(f"Model satellite has landed or impacted ground ({alt:.1f} m).")
            elif vz < Config.CRITICAL_DESCENT_RATE_MPS:
                status = OrbitStatus.CRITICAL_REENTRY
                severity = "critical"
                reason.append(f"Rapid terminal descent ({vz:.1f} m/s). Parachute deploy / recovery check required.")
            elif vz < Config.MODERATE_DESCENT_RATE_MPS:
                status = OrbitStatus.ORBITAL_DECAY
                severity = "warning"
                reason.append(f"Descent phase active ({vz:.1f} m/s).")
            else:
                status = OrbitStatus.STABLE_ORBIT
                severity = "nominal"
                reason.append(f"Model satellite airborne / stable test state at {alt:.1f} m.")

        # 3. Estimated Time to De-Orbit / Re-entry
        est_time_to_reentry_sec = None
        if vz < -0.05 and alt > min_boundary:
            delta_h = max(0.0, alt - min_boundary)
            est_time_to_reentry_sec = round(delta_h / abs(vz), 1)

        # 4. Heading Trajectory Analysis
        # Check if satellite is spin-stabilized or chaotically tumbling
        heading_delta = (next_heading - curr_heading + 180.0) % 360.0 - 180.0
        spin_stability = "STABLE"
        if abs(heading_rate) > 40.0:
            spin_stability = "HIGH_RATE_SPIN"
        elif is_tumbling:
            spin_stability = "CHAOTIC_TUMBLE"
        elif abs(heading_rate) > 10.0:
            spin_stability = "MODERATE_DRIFT"

        # 5. Stability Health Score (0 - 100%)
        # Deduct for low altitude, descent rate, pressure rise, and tumbling
        score = 100.0
        if self.mission_mode == "ORBITAL":
            if alt < nominal_target:
                alt_penalty = min(50.0, (nominal_target - alt) / 1000.0)
                score -= alt_penalty
        else:
            if alt < min_boundary:
                score = 0.0

        if vz < 0.0:
            vz_penalty = min(30.0, abs(vz) * 0.5)
            score -= vz_penalty

        if is_tumbling:
            score -= 25.0
        elif attitude_unstable:
            score -= 10.0

        score = max(0.0, min(100.0, score))

        return {
            "is_in_orbit": is_in_orbit,
            "orbit_status": status,
            "severity": severity,
            "summary_message": " | ".join(reason),
            "stability_score": round(score, 1),
            "estimated_reentry_time_sec": est_time_to_reentry_sec,
            "spin_stability": spin_stability,
            "is_tumbling": is_tumbling,
            "tumble_rate_deg_s": tumble_rate,
            "current_heading_deg": curr_heading,
            "predicted_next_heading_deg": next_heading,
            "heading_delta_deg": round(heading_delta, 2),
            "decay_rate_mps": round(abs(vz) if vz < 0 else 0.0, 3)
        }

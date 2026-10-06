"""
Fused State Estimator for Model Satellite.
Orchestrates Attitude EKF and Altitude KF, rotating body-frame IMU accelerations
into the Earth/Inertial frame to produce accurate 6-DOF state estimations.
"""
import math
import numpy as np
from config import Config
from satellite.sensors import SensorDataPacket
from satellite.kalman.attitude_ekf import AttitudeEKF
from satellite.kalman.altitude_kf import AltitudeKalmanFilter

class FusedSatelliteEstimator:
    """
    Fused multi-sensor estimator combining:
    - Attitude EKF (MPU6050 Gyro + Accel + LSM6DSM Magnetometer)
    - Altitude KF (BMP280 Barometer + Attitude-rotated MPU6050 Accel)
    """
    def __init__(self,
                 mission_mode: str = Config.MISSION_MODE,
                 initial_altitude_m: float = 300000.0):
        self.mission_mode = mission_mode
        self.attitude_ekf = AttitudeEKF()
        self.altitude_kf = AltitudeKalmanFilter(initial_altitude_m=initial_altitude_m)
        self.step_count = 0
        self.last_timestamp = None

    def reset(self, initial_altitude_m: float = 300000.0):
        """Reset internal filter states."""
        self.attitude_ekf = AttitudeEKF()
        self.altitude_kf = AltitudeKalmanFilter(initial_altitude_m=initial_altitude_m)
        self.step_count = 0
        self.last_timestamp = None

    def process_packet(self, packet: SensorDataPacket, lookahead_sec: float = Config.PREDICTION_LOOKAHEAD_SEC) -> dict:
        """
        Processes a single normalized SensorDataPacket:
        1. Predict attitude using MPU6050 Gyro
        2. Correct attitude using MPU6050 Accel & LSM6DSM Mag
        3. Rotate body accel to NED frame to get true vertical inertial acceleration
        4. Predict & update Altitude KF with BMP280 altitude + vertical accel
        5. Predict NEXT HEADING and attitude stability
        """
        self.step_count += 1
        dt = packet.dt
        self.last_timestamp = packet.timestamp

        # 1 & 2: Attitude EKF predict & updates
        self.attitude_ekf.predict(packet.gyro_body, dt)
        self.attitude_ekf.update_accelerometer(packet.accel_body)
        self.attitude_ekf.update_magnetometer(packet.mag_body)

        # 3. Transform body acceleration into NED frame: a_ned = R * a_body
        R_b_to_n = self.attitude_ekf.rotation_matrix()
        a_ned = R_b_to_n @ packet.accel_body
        
        # In NED frame, Down is +Z. True vertical acceleration (upwards positive):
        # a_vertical_up = -(a_ned[2] - g)
        a_vertical_inertial = -(a_ned[2] - Config.GRAVITY_ACCEL)

        # 4. Altitude KF predict & update
        self.altitude_kf.predict(dt)
        self.altitude_kf.update(packet.altitude_raw_m, a_vertical_inertial)

        # 5. Heading & Attitude calculations
        roll, pitch, yaw = self.attitude_ekf.get_euler_angles()
        heading_prediction = self.attitude_ekf.predict_next_heading(lookahead_sec=lookahead_sec)
        
        current_heading_deg = heading_prediction["current_heading_deg"]
        next_heading_deg = heading_prediction["next_heading_deg"]
        heading_rate_deg_s = heading_prediction["heading_rate_deg_s"]
        tumble_rate_deg_s = heading_prediction["tumble_rate_deg_s"]

        # Aerodynamic drag estimate based on pressure & vertical velocity
        # Dynamic pressure q = 0.5 * rho * v^2
        # Atmospheric density approx from ideal gas law: rho = P / (R_air * T)
        t_kelvin = max(50.0, packet.temperature_c + 273.15)
        air_density = packet.pressure_pa / (Config.GAS_CONSTANT_AIR * t_kelvin)
        est_drag_accel = 0.5 * air_density * (self.altitude_kf.vertical_velocity ** 2) * 0.01  # normalized for unit area

        return {
            "timestamp": packet.timestamp,
            "step": self.step_count,
            "dt": round(dt, 4),
            "attitude": {
                "quaternion": [round(float(q), 5) for q in self.attitude_ekf.q],
                "roll_deg": round(math.degrees(roll), 2),
                "pitch_deg": round(math.degrees(pitch), 2),
                "yaw_deg": round(math.degrees(yaw), 2),
                "current_heading_deg": current_heading_deg,
                "next_heading_deg": next_heading_deg,
                "heading_rate_deg_s": heading_rate_deg_s,
                "gyro_bias_rad_s": [round(float(b), 6) for b in self.attitude_ekf.gyro_bias],
                "tumble_rate_deg_s": tumble_rate_deg_s,
                "trajectory": heading_prediction["trajectory"]
            },
            "altitude_state": {
                "filtered_altitude_m": round(self.altitude_kf.altitude, 2),
                "raw_bmp_altitude_m": round(packet.altitude_raw_m, 2),
                "vertical_velocity_mps": round(self.altitude_kf.vertical_velocity, 3),
                "vertical_accel_mps2": round(self.altitude_kf.vertical_acceleration, 3),
                "air_density_kg_m3": float(f"{air_density:.4e}"),
                "drag_accel_estimate_mps2": float(f"{est_drag_accel:.4e}")
            },
            "raw_sensors": packet.to_dict()
        }

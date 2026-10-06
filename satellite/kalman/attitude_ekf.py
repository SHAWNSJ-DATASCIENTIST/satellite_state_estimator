"""
Extended Kalman Filter (EKF) for 3D Satellite Attitude Determination.
Fuses MPU6050 Gyroscope (propagation) with MPU6050 Accelerometer and
LSM6DSM Magnetometer (observation updates).
Computes current attitude (quaternion, roll, pitch, yaw) and predicts the NEXT HEADING.
"""
import math
import numpy as np
from config import Config

class AttitudeEKF:
    """
    7-State Extended Kalman Filter for Attitude:
    x = [q0, q1, q2, q3, bgx, bgy, bgz]^T
    where q is unit quaternion [w, x, y, z], bg is gyro bias in rad/s.
    """
    def __init__(self,
                 mag_declination_deg: float = Config.MAG_DECLINATION_DEG,
                 q_gyro_noise: float = Config.GYRO_NOISE_STD,
                 q_bias_noise: float = Config.GYRO_BIAS_DRIFT_STD,
                 r_accel_noise: float = Config.ACCEL_NOISE_STD,
                 r_mag_noise: float = Config.MAG_NOISE_STD):
        
        self.mag_declination_rad = math.radians(mag_declination_deg)
        
        # State vector: [q0, q1, q2, q3, bgx, bgy, bgz]
        # Initial identity quaternion (level, facing north) and zero biases
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        
        # State Covariance matrix P (7x7)
        self.P = np.eye(7, dtype=float) * 0.01
        self.P[4:7, 4:7] = np.eye(3) * 0.001  # Bias uncertainty
        
        # Process Noise Q (7x7)
        self.Q = np.zeros((7, 7), dtype=float)
        self.q_gyro_var = q_gyro_noise ** 2
        self.q_bias_var = q_bias_noise ** 2
        self.Q[0:4, 0:4] = np.eye(4) * self.q_gyro_var
        self.Q[4:7, 4:7] = np.eye(3) * self.q_bias_var
        
        # Measurement Noise R
        self.r_acc_var = r_accel_noise ** 2
        self.r_mag_var = r_mag_noise ** 2
        
        # Normalized Earth magnetic field reference in NED frame [bx, 0, bz]
        mag_ned_raw = Config.DEFAULT_MAG_FIELD_NED
        self.mag_ref_ned = mag_ned_raw / np.linalg.norm(mag_ned_raw)
        
        self.last_omega = np.zeros(3)
        self.initialized = False

    @property
    def q(self) -> np.ndarray:
        """Normalized unit quaternion [q0, q1, q2, q3]."""
        return self.x[0:4]

    @property
    def gyro_bias(self) -> np.ndarray:
        """Estimated gyroscope bias in rad/s [bgx, bgy, bgz]."""
        return self.x[4:7]

    def _normalize_quaternion(self):
        """Ensures quaternion remains on unit sphere."""
        q_norm = np.linalg.norm(self.x[0:4])
        if q_norm > 1e-9:
            self.x[0:4] /= q_norm
        else:
            self.x[0:4] = np.array([1.0, 0.0, 0.0, 0.0])

    def rotation_matrix(self) -> np.ndarray:
        """
        Direction Cosine Matrix (Body to NED frame).
        R_b_to_n transforms a vector in body frame into NED frame.
        """
        q0, q1, q2, q3 = self.x[0:4]
        return np.array([
            [1.0 - 2.0*(q2*q2 + q3*q3), 2.0*(q1*q2 - q0*q3),       2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3),       1.0 - 2.0*(q1*q1 + q3*q3), 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2),       2.0*(q2*q3 + q0*q1),       1.0 - 2.0*(q1*q1 + q2*q2)]
        ], dtype=float)

    def get_euler_angles(self) -> tuple[float, float, float]:
        """
        Extracts Roll (phi), Pitch (theta), Yaw/Heading (psi) in radians.
        Using aerospace NED Z-Y-X convention.
        """
        q0, q1, q2, q3 = self.x[0:4]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation) - clamped to prevent numerical singularity
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation / Heading relative to true North)
        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny_cosp, cosy_cosp) + self.mag_declination_rad
        
        # Wrap yaw to [-pi, pi]
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        
        return roll, pitch, yaw

    def get_heading_degrees(self) -> float:
        """Returns current heading compass angle in [0, 360) degrees."""
        _, _, yaw = self.get_euler_angles()
        heading_deg = math.degrees(yaw)
        return (heading_deg + 360.0) % 360.0

    def predict(self, gyro_body: np.ndarray, dt: float):
        """
        EKF Prediction step using angular rate from MPU6050 Gyroscope.
        gyro_body: [gx, gy, gz] in rad/s
        dt: time step in seconds
        """
        if dt <= 0.0:
            return

        # Unbiased angular velocity
        omega = gyro_body - self.x[4:7]
        self.last_omega = omega
        wx, wy, wz = omega

        q0, q1, q2, q3 = self.x[0:4]

        # Quaternion derivative matrix Omega
        # q_dot = 0.5 * Omega * q
        Omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0,  wx],
            [wz,  wy, -wx,  0.0]
        ], dtype=float)

        # Propagate quaternion
        q_new = self.x[0:4] + 0.5 * Omega @ self.x[0:4] * dt
        self.x[0:4] = q_new
        self._normalize_quaternion()

        # State transition Jacobian F (7x7)
        F = np.eye(7, dtype=float)
        F[0:4, 0:4] += 0.5 * Omega * dt

        # Partial derivative of q_dot w.r.t gyro bias: -0.5 * Xi(q) * dt
        Xi = np.array([
            [-q1, -q2, -q3],
            [ q0, -q3,  q2],
            [ q3,  q0, -q1],
            [-q2,  q1,  q0]
        ], dtype=float)
        F[0:4, 4:7] = -0.5 * Xi * dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_accelerometer(self, accel_body: np.ndarray):
        """
        EKF Measurement update from MPU6050 Accelerometer.
        Estimates Roll and Pitch relative to the gravity vector.
        accel_body: [ax, ay, az] in m/s^2
        """
        acc_norm = np.linalg.norm(accel_body)
        if acc_norm < 1e-4:
            return

        # Normalized measured acceleration
        z_acc = accel_body / acc_norm

        # If satellite experiences massive dynamic acceleration (e.g. thrust / impact / spin > 2g),
        # dynamically increase measurement variance to prevent corruption
        accel_diff = abs(acc_norm - Config.GRAVITY_ACCEL)
        r_scale = 1.0 + (accel_diff / 5.0) ** 2
        R_acc = np.eye(3) * (self.r_acc_var * r_scale)

        q0, q1, q2, q3 = self.x[0:4]

        # Predicted gravity in body frame (for NED down vector [0, 0, 1])
        # h(x) = R_n_to_b * [0, 0, 1]^T
        h_acc = np.array([
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q0 * q1 + q2 * q3),
            q0*q0 - q1*q1 - q2*q2 + q3*q3
        ], dtype=float)

        # Measurement Jacobian H_acc (3x7) w.r.t [q0, q1, q2, q3, bgx, bgy, bgz]
        H_acc = np.zeros((3, 7), dtype=float)
        H_acc[0, 0:4] = [-2.0 * q2,  2.0 * q3, -2.0 * q0,  2.0 * q1]
        H_acc[1, 0:4] = [ 2.0 * q1,  2.0 * q0,  2.0 * q3,  2.0 * q2]
        H_acc[2, 0:4] = [ 2.0 * q0, -2.0 * q1, -2.0 * q2,  2.0 * q3]

        # Innovation
        y = z_acc - h_acc
        S = H_acc @ self.P @ H_acc.T + R_acc
        K = self.P @ H_acc.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H_acc) @ self.P
        self._normalize_quaternion()

    def update_magnetometer(self, mag_body: np.ndarray):
        """
        EKF Measurement update from LSM6DSM Magnetometer.
        Observes Heading / Yaw relative to Earth's geomagnetic field.
        mag_body: [mx, my, mz] in uT
        """
        mag_norm = np.linalg.norm(mag_body)
        if mag_norm < 1e-3:
            return

        z_mag = mag_body / mag_norm

        q0, q1, q2, q3 = self.x[0:4]
        R_b_to_n = self.rotation_matrix()

        # Rotate body magnetic reading into world frame to estimate local horizontal / vertical field
        h_world = R_b_to_n @ z_mag
        bx = math.sqrt(h_world[0]*h_world[0] + h_world[1]*h_world[1])
        bz = h_world[2]

        # Expected body magnetic field from Earth reference [bx, 0, bz]
        h_mag = np.array([
            bx * (q0*q0 + q1*q1 - q2*q2 - q3*q3) + 2.0 * bz * (q1*q3 - q0*q2),
            2.0 * bx * (q1*q2 - q0*q3)          + 2.0 * bz * (q0*q1 + q2*q3),
            2.0 * bx * (q0*q2 + q1*q3)          + bz * (q0*q0 - q1*q1 - q2*q2 + q3*q3)
        ], dtype=float)

        # Measurement Jacobian H_mag (3x7)
        H_mag = np.zeros((3, 7), dtype=float)
        H_mag[0, 0:4] = [
             2.0*bx*q0 - 2.0*bz*q2,
             2.0*bx*q1 + 2.0*bz*q3,
            -2.0*bx*q2 - 2.0*bz*q0,
            -2.0*bx*q3 + 2.0*bz*q1
        ]
        H_mag[1, 0:4] = [
            -2.0*bx*q3 + 2.0*bz*q1,
             2.0*bx*q2 + 2.0*bz*q0,
             2.0*bx*q1 + 2.0*bz*q3,
            -2.0*bx*q0 + 2.0*bz*q2
        ]
        H_mag[2, 0:4] = [
             2.0*bx*q2 + 2.0*bz*q0,
             2.0*bx*q3 - 2.0*bz*q1,
             2.0*bx*q0 - 2.0*bz*q2,
             2.0*bx*q1 + 2.0*bz*q3
        ]

        R_mag = np.eye(3) * self.r_mag_var

        # Innovation
        y = z_mag - h_mag
        S = H_mag @ self.P @ H_mag.T + R_mag
        K = self.P @ H_mag.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H_mag) @ self.P
        self._normalize_quaternion()

    def predict_next_heading(self, lookahead_sec: float = Config.PREDICTION_LOOKAHEAD_SEC) -> dict:
        """
        Calculates the NEXT HEADING after lookahead_sec seconds based on
        current attitude and angular velocity.
        
        Using Euler kinematic relationship:
        dot(psi) = (omega_y * sin(phi) + omega_z * cos(phi)) / cos(theta)
        """
        roll, pitch, yaw = self.get_euler_angles()
        current_heading_deg = (math.degrees(yaw) + 360.0) % 360.0
        
        wx, wy, wz = self.last_omega
        
        # Guard against gimbal lock singularity at pitch = +/- 90 deg
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) < 1e-3:
            cos_pitch = 1e-3 if cos_pitch >= 0 else -1e-3
            
        # Yaw rate in rad/s
        yaw_rate_rad_s = (wy * math.sin(roll) + wz * math.cos(roll)) / cos_pitch
        yaw_rate_deg_s = math.degrees(yaw_rate_rad_s)
        
        # Predicted heading at t + lookahead_sec
        predicted_yaw_deg = current_heading_deg + yaw_rate_deg_s * lookahead_sec
        next_heading_deg = (predicted_yaw_deg + 360.0) % 360.0
        
        # Calculate full projected heading trajectory [1s, 2s, 5s, 10s]
        trajectory = []
        for dt_step in [0.5, 1.0, 2.0, 5.0, 10.0]:
            projected = (current_heading_deg + yaw_rate_deg_s * dt_step + 360.0) % 360.0
            trajectory.append({
                "delta_sec": dt_step,
                "projected_heading_deg": round(projected, 2)
            })
            
        # Tumble rate = magnitude of 3D angular velocity vector
        tumble_rate_deg_s = math.degrees(float(np.linalg.norm(self.last_omega)))
        
        return {
            "current_heading_deg": round(current_heading_deg, 2),
            "next_heading_deg": round(next_heading_deg, 2),
            "heading_rate_deg_s": round(yaw_rate_deg_s, 3),
            "lookahead_sec": lookahead_sec,
            "tumble_rate_deg_s": round(tumble_rate_deg_s, 2),
            "trajectory": trajectory
        }

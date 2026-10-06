"""
Kalman Filter for Satellite Altitude, Vertical Velocity, and Decay Rate.
Fuses BMP280 pressure-derived altitude with MPU6050 vertical inertial acceleration.
Provides filtered altitude and vertical velocity (dh/dt) to detect orbital decay.
"""
import numpy as np
from config import Config

class AltitudeKalmanFilter:
    """
    3-State Linear Kalman Filter for vertical kinematics:
    x = [h, vz, az]^T
    h: altitude (m)
    vz: vertical velocity (m/s) [decay rate when negative]
    az: vertical acceleration (m/s^2)
    """
    def __init__(self,
                 initial_altitude_m: float = 300000.0,
                 r_altitude_std: float = Config.ALTITUDE_NOISE_STD,
                 r_accel_std: float = Config.ACCEL_NOISE_STD):
        
        # State vector [h, vz, az]
        self.x = np.array([initial_altitude_m, 0.0, 0.0], dtype=float)
        
        # Covariance matrix P (3x3)
        self.P = np.diag([25.0, 1.0, 0.5])
        
        # Measurement noise covariance R (2x2: [h_baro, a_inertial])
        self.R = np.diag([r_altitude_std ** 2, r_accel_std ** 2])
        
        # Process noise spectral density q_w (acceleration jerk)
        self.q_jerk = 0.5
        self.initialized = False

    @property
    def altitude(self) -> float:
        return float(self.x[0])

    @property
    def vertical_velocity(self) -> float:
        return float(self.x[1])

    @property
    def vertical_acceleration(self) -> float:
        return float(self.x[2])

    def predict(self, dt: float):
        """
        State propagation over delta time dt.
        F = [[1, dt, 0.5*dt^2],
             [0,  1, dt],
             [0,  0, 1]]
        """
        if dt <= 0:
            return
            
        dt2 = 0.5 * dt * dt
        F = np.array([
            [1.0, dt,  dt2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        
        # Discrete process noise matrix Q
        # Derived from continuous white noise on derivative of acceleration
        dt3 = dt * dt2 / 3.0
        dt4 = 0.25 * dt * dt * dt * dt
        dt5 = 0.05 * dt ** 5
        Q = self.q_jerk * np.array([
            [dt5,   0.125 * dt**4, dt3],
            [0.125 * dt**4, dt3,   dt2],
            [dt3,   dt2,           dt]
        ], dtype=float)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, altitude_raw_m: float, vertical_accel_mps2: float):
        """
        Measurement update fusing BMP280 altitude and inertial acceleration.
        z = [h_bmp, a_inertial]^T
        H = [[1, 0, 0],
             [0, 0, 1]]
        """
        if not self.initialized:
            self.x[0] = altitude_raw_m
            self.x[1] = 0.0
            self.x[2] = vertical_accel_mps2
            self.P = np.diag([self.R[0, 0], 1.0, self.R[1, 1]])
            self.initialized = True
            return
            
        z = np.array([altitude_raw_m, vertical_accel_mps2], dtype=float)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        
        # Innovation y = z - Hx
        y = z - H @ self.x
        
        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State & covariance update
        self.x = self.x + K @ y
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

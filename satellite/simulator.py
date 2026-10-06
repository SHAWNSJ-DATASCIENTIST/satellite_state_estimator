"""
Telemetry Simulator for Model / Space Satellite.
Generates realistic sensor readings (BMP280, MPU6050, LSM6DSM, RTC) for testing
Kalman filter convergence, heading predictions, and orbit decay scenarios.
"""
import math
import random
import time
import numpy as np
from config import Config

class SatelliteFlightSimulator:
    """Simulates 6-DOF satellite physics and sensor measurements."""
    
    def __init__(self, scenario: str = "stable_orbit"):
        self.scenario = scenario
        self.reset(scenario)

    def reset(self, scenario: str = None):
        if scenario:
            self.scenario = scenario
            
        self.t = 0.0
        self.rtc_base = time.time()
        self.dt = 0.1  # 10 Hz simulation step
        
        # Scenario configurations
        if self.scenario == "stable_orbit":
            self.altitude = 350000.0  # 350 km LEO
            self.vz = -0.05           # Very slow nominal decay
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 45.0           # 45 deg initial heading
            self.omega = np.array([0.005, 0.002, 0.015])  # rad/s slow spin stabilization
        elif self.scenario == "orbital_decay":
            self.altitude = 135000.0  # 135 km VLEO
            self.vz = -15.0           # Noticeable decay
            self.roll = 5.0
            self.pitch = -3.0
            self.yaw = 120.0
            self.omega = np.array([0.08, -0.05, 0.12])   # Noticeable attitude drift
        elif self.scenario == "critical_reentry":
            self.altitude = 85000.0   # 85 km (below Karman line)
            self.vz = -95.0           # Plunging
            self.roll = 45.0
            self.pitch = -30.0
            self.yaw = 280.0
            self.omega = np.array([1.2, -1.8, 2.5])      # Chaotic high-rate tumbling
        elif self.scenario == "suborbital_cansat":
            self.altitude = 2500.0    # 2.5 km CanSat apogee
            self.vz = -8.0            # Parachute descent
            self.roll = 2.0
            self.pitch = 1.0
            self.yaw = 90.0
            self.omega = np.array([0.02, 0.01, 0.05])
        else:
            self.altitude = 250000.0
            self.vz = -0.1
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.omega = np.array([0.01, 0.01, 0.01])

        # Persistent gyro bias
        self.gyro_bias = np.array([0.002, -0.0015, 0.003])

    def step(self) -> dict:
        """Advances simulation by dt and generates next raw sensor packet."""
        self.t += self.dt
        current_rtc = self.rtc_base + self.t
        
        # 1. Update Kinematics & Attitude
        self.altitude += self.vz * self.dt
        
        # Scenarios evolve over time
        if self.scenario == "orbital_decay":
            self.vz -= 0.1 * self.dt  # Drag increases descent
            self.omega += np.random.normal(0, 0.005, 3)
        elif self.scenario == "critical_reentry":
            self.vz -= 0.5 * self.dt
            # Rapid tumbling amplification
            self.omega += np.sin(self.t) * 0.05
        elif self.scenario == "suborbital_cansat":
            if self.altitude < 0:
                self.altitude = 0.0
                self.vz = 0.0
                self.omega = np.zeros(3)

        # Update Euler angles from angular rate (simple kinematic integration)
        phi = math.radians(self.roll)
        theta = math.radians(self.pitch)
        cos_theta = max(0.01, abs(math.cos(theta)))
        
        d_roll = self.omega[0] + (self.omega[1]*math.sin(phi) + self.omega[2]*math.cos(phi)) * math.tan(theta)
        d_pitch = self.omega[1]*math.cos(phi) - self.omega[2]*math.sin(phi)
        d_yaw = (self.omega[1]*math.sin(phi) + self.omega[2]*math.cos(phi)) / cos_theta
        
        self.roll += math.degrees(d_roll * self.dt)
        self.pitch += math.degrees(d_pitch * self.dt)
        self.yaw = (self.yaw + math.degrees(d_yaw * self.dt)) % 360.0

        # 2. Simulate BMP280 Pressure & Temperature
        # Pressure model based on altitude:
        if self.altitude > 180000.0:
            # Thermosphere / LEO vacuum regime
            pressure_pa = 1e-6 * math.exp(-(self.altitude - 180000.0) / 25000.0)
            temp_c = -45.0 + 5.0 * math.sin(self.t * 0.1)
        elif self.altitude > 32000.0:
            # Mesosphere / Upper Stratosphere
            pressure_pa = 1000.0 * math.exp(-(self.altitude - 32000.0) / 8500.0)
            temp_c = -50.0 + 10.0 * math.sin(self.t * 0.05)
        else:
            # Troposphere / Lower Stratosphere: standard barometric formula inverse
            ratio = max(0.0, 1.0 - (self.altitude / 44330.77))
            pressure_pa = Config.SEA_LEVEL_PRESSURE_PA * math.pow(ratio, 5.25588)
            temp_c = 15.0 - (0.0065 * self.altitude)
            
        # Add BMP280 sensor noise proportional to pressure scale
        noise_std = max(1e-9, min(1.5, 0.015 * pressure_pa))
        pressure_pa += random.gauss(0, noise_std)
        pressure_pa = max(1e-12, pressure_pa)

        # 3. Simulate MPU6050 Accelerometer & Gyroscope
        # In freefall / orbit, accelerometer measures primarily non-gravitational specific forces (drag + noise)
        # Plus centripetal/vibrational components
        r_rad = math.radians(self.roll)
        p_rad = math.radians(self.pitch)
        y_rad = math.radians(self.yaw)
        
        # Rotation matrix from NED to Body frame: R_n_to_b = (R_b_to_n)^T
        # We project the apparent gravity vector [0, 0, g] into body frame
        # If in orbit (microgravity free-fall), apparent gravity is cancelled by orbital centrifugal force,
        # but CanSat / atmospheric test experience gravity
        if self.scenario == "suborbital_cansat" or self.altitude < 80000.0:
            # Significant apparent gravity / atmospheric deceleration
            g_body_x = -Config.GRAVITY_ACCEL * math.sin(p_rad)
            g_body_y =  Config.GRAVITY_ACCEL * math.sin(r_rad) * math.cos(p_rad)
            g_body_z =  Config.GRAVITY_ACCEL * math.cos(r_rad) * math.cos(p_rad)
        else:
            # Orbital freefall (microgravity < 1e-4 g) + slight atmospheric drag
            drag_val = 0.001 * (1.0 + random.uniform(-0.1, 0.1))
            g_body_x = 0.0
            g_body_y = 0.0
            g_body_z = drag_val

        # Add accelerometer noise
        ax = g_body_x + random.gauss(0, 0.08)
        ay = g_body_y + random.gauss(0, 0.08)
        az = g_body_z + random.gauss(0, 0.08)

        # Gyroscope measurement with bias and noise
        gx = self.omega[0] + self.gyro_bias[0] + random.gauss(0, 0.005)
        gy = self.omega[1] + self.gyro_bias[1] + random.gauss(0, 0.005)
        gz = self.omega[2] + self.gyro_bias[2] + random.gauss(0, 0.005)

        # 4. Simulate LSM6DSM Magnetometer
        # Earth magnetic field in NED frame
        mag_ned = Config.DEFAULT_MAG_FIELD_NED
        # Rotate mag_ned to body frame: m_body = R_b_to_n^T * m_ned
        # For simplified projection with yaw, pitch, roll:
        cy, sy = math.cos(y_rad), math.sin(y_rad)
        cp, sp = math.cos(p_rad), math.sin(p_rad)
        cr, sr = math.cos(r_rad), math.sin(r_rad)
        
        # Body frame components
        mx = mag_ned[0]*(cy*cp) + mag_ned[1]*(sy*cp) - mag_ned[2]*sp
        my = mag_ned[0]*(cy*sp*sr - sy*cr) + mag_ned[1]*(sy*sp*sr + cy*cr) + mag_ned[2]*(cp*sr)
        mz = mag_ned[0]*(cy*sp*cr + sy*sr) + mag_ned[1]*(sy*sp*cr - cy*sr) + mag_ned[2]*(cp*cr)
        
        # Add magnetometer noise
        mx += random.gauss(0, 0.3)
        my += random.gauss(0, 0.3)
        mz += random.gauss(0, 0.3)

        return {
            "rtc_time": current_rtc,
            "bmp280": {
                "pressure_pa": round(pressure_pa, 3),
                "temperature_c": round(temp_c, 2)
            },
            "mpu6050": {
                "accel_x": round(ax, 4),
                "accel_y": round(ay, 4),
                "accel_z": round(az, 4),
                "gyro_x": round(gx, 5),
                "gyro_y": round(gy, 5),
                "gyro_z": round(gz, 5)
            },
            "lsm6dsm": {
                "mag_x": round(mx, 3),
                "mag_y": round(my, 3),
                "mag_z": round(mz, 3)
            }
        }

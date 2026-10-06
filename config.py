"""
Configuration parameters for the Satellite State Estimation & Kalman Filter system.
Contains sensor noise characteristics, physical constants, filter covariances,
and orbital boundary thresholds.
"""
import os
import numpy as np

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'orbital-kalman-telemetry-secret-key-2026')
    DEBUG = False
    
    # Mission Mode: 'ORBITAL' (LEO / Space satellite) or 'SUBORBITAL_TEST' (CanSat / Model rocket / High-alt balloon)
    MISSION_MODE = os.environ.get('MISSION_MODE', 'ORBITAL')
    
    # Atmospheric & Orbital Constants
    SEA_LEVEL_PRESSURE_PA = 101325.0       # Standard sea level pressure (Pa)
    SEA_LEVEL_TEMP_K = 288.15              # Sea level standard temperature (K)
    TEMP_LAPSE_RATE = 0.0065               # K/m (troposphere)
    GRAVITY_ACCEL = 9.80665                # m/s^2 standard gravity
    GAS_CONSTANT_AIR = 287.05              # J/(kg*K)
    EARTH_RADIUS_M = 6371000.0             # Earth mean radius in meters
    
    # Orbit Determination Thresholds
    # In ORBITAL mode: Kármán line (100 km) is the nominal edge of space. Below this, atmosphere rapidly de-orbits.
    KARMAN_LINE_ALTITUDE_M = 100000.0      # 100 km
    LEO_MIN_STABLE_ALTITUDE_M = 160000.0   # 160 km (minimum short-term orbit without immediate burnup)
    
    # In SUBORBITAL_TEST mode (for model satellites / CanSat testing):
    SUBORBITAL_CEILING_M = 4000.0          # 4 km model ceiling
    SUBORBITAL_FLOOR_M = 50.0              # Ground impact threshold
    
    # Orbit Decay & Tumbling Thresholds
    CRITICAL_DESCENT_RATE_MPS = -25.0      # Descent rate indicating rapid de-orbit (m/s)
    MODERATE_DESCENT_RATE_MPS = -2.0       # Descent rate indicating active orbital decay (m/s)
    TUMBLE_THRESHOLD_DEG_S = 45.0          # Angular rate indicating satellite is tumbling uncontrolled
    MODERATE_TUMBLE_DEG_S = 15.0           # Angular rate indicating attitude instability
    
    # Geomagnetic Field Reference (Local or Mean LEO field in microTeslas, NED frame)
    # Earth magnetic field at LEO ~ 30-50 uT
    MAG_DECLINATION_DEG = 0.0              # Magnetic declination (deg) - offset from true north
    DEFAULT_MAG_FIELD_NED = np.array([22.0, 0.0, 42.0])  # [North, East, Down] in uT
    
    # MPU6050 IMU Sensor Noise Covariance Defaults
    ACCEL_NOISE_STD = 0.15                 # m/s^2 accelerometer standard deviation
    GYRO_NOISE_STD = 0.02                  # rad/s gyroscope noise standard deviation
    GYRO_BIAS_DRIFT_STD = 0.0005           # rad/s^2 gyro bias random walk
    
    # LSM6DSM Magnetometer Noise Covariance
    MAG_NOISE_STD = 0.8                    # uT magnetometer standard deviation
    
    # BMP280 Barometer Noise Covariance
    PRESSURE_NOISE_STD = 12.0              # Pa pressure noise standard deviation
    ALTITUDE_NOISE_STD = 1.5               # m equivalent altitude noise standard deviation
    
    # Heading Prediction
    PREDICTION_LOOKAHEAD_SEC = 1.0         # Default next heading prediction horizon (seconds)
    
    # Web / API Settings
    MAX_TELEMETRY_HISTORY = 500            # In-memory points kept for charts

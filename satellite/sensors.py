"""
Sensor processing and data parsing for BMP280, MPU6050, LSM6DSM, and RTC.
Standardizes telemetry into SI units for the Kalman Filter pipelines.
"""
import math
import datetime
import numpy as np
from config import Config

class SensorDataPacket:
    """Represents a validated, normalized telemetry packet."""
    def __init__(self,
                 timestamp: float,
                 dt: float,
                 pressure_pa: float,
                 temperature_c: float,
                 altitude_raw_m: float,
                 accel_body: np.ndarray,      # [ax, ay, az] in m/s^2
                 gyro_body: np.ndarray,       # [gx, gy, gz] in rad/s
                 mag_body: np.ndarray):       # [mx, my, mz] in uT
        self.timestamp = timestamp
        self.dt = dt
        self.pressure_pa = pressure_pa
        self.temperature_c = temperature_c
        self.altitude_raw_m = altitude_raw_m
        self.accel_body = accel_body
        self.gyro_body = gyro_body
        self.mag_body = mag_body

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "dt": round(self.dt, 4),
            "bmp280": {
                "pressure_pa": round(self.pressure_pa, 2),
                "temperature_c": round(self.temperature_c, 2),
                "altitude_raw_m": round(self.altitude_raw_m, 2)
            },
            "mpu6050": {
                "accel_x": round(float(self.accel_body[0]), 4),
                "accel_y": round(float(self.accel_body[1]), 4),
                "accel_z": round(float(self.accel_body[2]), 4),
                "gyro_x_rad_s": round(float(self.gyro_body[0]), 5),
                "gyro_y_rad_s": round(float(self.gyro_body[1]), 5),
                "gyro_z_rad_s": round(float(self.gyro_body[2]), 5),
                "gyro_x_deg_s": round(float(math.degrees(self.gyro_body[0])), 3),
                "gyro_y_deg_s": round(float(math.degrees(self.gyro_body[1])), 3),
                "gyro_z_deg_s": round(float(math.degrees(self.gyro_body[2])), 3)
            },
            "lsm6dsm": {
                "mag_x_uT": round(float(self.mag_body[0]), 3),
                "mag_y_uT": round(float(self.mag_body[1]), 3),
                "mag_z_uT": round(float(self.mag_body[2]), 3),
                "mag_norm_uT": round(float(np.linalg.norm(self.mag_body)), 3)
            }
        }


class SensorParser:
    """Parses raw input payloads from BMP280, MPU6050, LSM6DSM, and RTC."""
    
    def __init__(self, p0: float = Config.SEA_LEVEL_PRESSURE_PA):
        self.p0 = p0
        self.last_timestamp = None
        
        # Hard iron bias and soft iron matrix for LSM6DSM magnetometer
        self.mag_hard_iron = np.array([0.0, 0.0, 0.0])
        self.mag_soft_iron = np.eye(3)

    def set_mag_calibration(self, hard_iron_offset: list, soft_iron_matrix: list = None):
        """Configure hard iron offset and optional soft iron matrix."""
        self.mag_hard_iron = np.array(hard_iron_offset, dtype=float)
        if soft_iron_matrix is not None:
            self.mag_soft_iron = np.array(soft_iron_matrix, dtype=float)

    def parse_rtc_time(self, raw_time) -> float:
        """Converts RTC input (Unix float, int, or ISO string) to Unix epoch float."""
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        elif isinstance(raw_time, str):
            try:
                # Try parsing ISO 8601 string
                dt = datetime.datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                return dt.timestamp()
            except ValueError:
                return float(raw_time)
        else:
            raise ValueError(f"Unsupported RTC time format: {type(raw_time)}")

    def pressure_to_altitude(self, pressure_pa: float, temp_c: float = 15.0) -> float:
        """
        Converts barometric pressure to altitude.
        Handles both tropospheric barometric regime and extreme low-pressure / space regime.
        - In lower atmosphere (P > 1000 Pa, up to ~30 km): Barometric formula.
        - In upper atmosphere / near-space (P <= 1000 Pa to near 0 Pa): Exponential density / scale height model.
        """
        if pressure_pa <= 0.0:
            # High vacuum / LEO orbital vacuum (> 200 km)
            return 300000.0  # 300 km nominal vacuum altitude
        
        # If pressure is greater than 1000 Pa (0 to ~31,000 meters)
        if pressure_pa > 1000.0:
            # Standard hypsometric formula
            ratio = pressure_pa / self.p0
            if ratio > 1.2:
                ratio = 1.2
            alt = 44330.77 * (1.0 - math.pow(ratio, 0.190284))
            return max(-500.0, alt)
        else:
            # High altitude / mesosphere / thermosphere scale height approximation
            # At 32 km (~1000 Pa), scale height H ~ 7.5 km in stratosphere/mesosphere,
            # expanding up to 25 km in thermosphere/LEO.
            p_clamped = max(1e-12, pressure_pa)
            if p_clamped < 1e-6:
                # High vacuum regime (LEO > 200 km)
                # ln(1e-6 / 1000) ~ -20.7
                alt = 180000.0 - 25000.0 * math.log(p_clamped / 1e-6)
            else:
                alt = 32000.0 - 8500.0 * math.log(p_clamped / 1000.0)
            return min(alt, 650000.0)

    def parse_payload(self, data: dict) -> SensorDataPacket:
        """
        Validates and parses raw dictionary payload from API or telemetry line.
        Expected keys:
        - rtc_time / timestamp
        - bmp280: {pressure_pa / pressure_hpa, temperature_c}
        - mpu6050: {accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z} (units can be specified or defaulted to m/s^2 and rad/s or deg/s)
        - lsm6dsm: {mag_x, mag_y, mag_z} (in uT)
        """
        if not isinstance(data, dict):
            raise ValueError("Payload must be a JSON object / dict")

        # 1. RTC Time & Delta Time
        raw_rtc = data.get('rtc_time', data.get('timestamp'))
        if raw_rtc is None:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            current_time = now
        else:
            current_time = self.parse_rtc_time(raw_rtc)

        if self.last_timestamp is None:
            dt = 0.05  # Default 20 Hz initial step
        else:
            dt = current_time - self.last_timestamp
            # Clamp dt to reasonable bounds (e.g. 1ms to 2.0s) to prevent Kalman divergence on gaps
            if dt <= 0:
                dt = 0.02
            elif dt > 2.0:
                dt = 0.1
        self.last_timestamp = current_time

        # 2. BMP280 Barometer
        bmp_data = data.get('bmp280', {})
        if 'pressure_pa' in bmp_data:
            pressure_pa = float(bmp_data['pressure_pa'])
        elif 'pressure_hpa' in bmp_data:
            pressure_pa = float(bmp_data['pressure_hpa']) * 100.0
        elif 'pressure' in bmp_data:
            p_val = float(bmp_data['pressure'])
            # Autodetect if given in hPa (e.g. ~1013) or Pa (e.g. ~101325)
            pressure_pa = p_val * 100.0 if p_val < 2000.0 else p_val
        else:
            pressure_pa = Config.SEA_LEVEL_PRESSURE_PA

        temp_c = float(bmp_data.get('temperature_c', bmp_data.get('temp', 15.0)))
        altitude_raw = self.pressure_to_altitude(pressure_pa, temp_c)

        # 3. MPU6050 IMU
        mpu_data = data.get('mpu6050', {})
        ax = float(mpu_data.get('accel_x', mpu_data.get('ax', 0.0)))
        ay = float(mpu_data.get('accel_y', mpu_data.get('ay', 0.0)))
        az = float(mpu_data.get('accel_z', mpu_data.get('az', 9.81)))

        # Autodetect if accel is in g units (magnitude ~ 1.0) or m/s^2 (magnitude ~ 9.8)
        acc_mag = math.sqrt(ax*ax + ay*ay + az*az)
        accel_units = mpu_data.get('accel_units', 'auto')
        if accel_units == 'g' or (accel_units == 'auto' and acc_mag > 0.05 and acc_mag < 2.5):
            # Scale from g to m/s^2
            ax *= Config.GRAVITY_ACCEL
            ay *= Config.GRAVITY_ACCEL
            az *= Config.GRAVITY_ACCEL

        gx = float(mpu_data.get('gyro_x', mpu_data.get('gx', 0.0)))
        gy = float(mpu_data.get('gyro_y', mpu_data.get('gy', 0.0)))
        gz = float(mpu_data.get('gyro_z', mpu_data.get('gz', 0.0)))

        # Autodetect if gyro is in deg/s or rad/s
        gyro_units = mpu_data.get('gyro_units', 'auto')
        if gyro_units == 'deg_s' or (gyro_units == 'auto' and max(abs(gx), abs(gy), abs(gz)) > 6.28):
            gx = math.radians(gx)
            gy = math.radians(gy)
            gz = math.radians(gz)

        accel_body = np.array([ax, ay, az], dtype=float)
        gyro_body = np.array([gx, gy, gz], dtype=float)

        # 4. LSM6DSM Magnetometer
        lsm_data = data.get('lsm6dsm', data.get('magnetometer', {}))
        mx = float(lsm_data.get('mag_x', lsm_data.get('mx', 20.0)))
        my = float(lsm_data.get('mag_y', lsm_data.get('my', 0.0)))
        mz = float(lsm_data.get('mag_z', lsm_data.get('mz', 40.0)))

        mag_raw = np.array([mx, my, mz], dtype=float)
        # Apply calibration: soft_iron * (mag_raw - hard_iron)
        mag_calibrated = self.mag_soft_iron @ (mag_raw - self.mag_hard_iron)

        return SensorDataPacket(
            timestamp=current_time,
            dt=dt,
            pressure_pa=pressure_pa,
            temperature_c=temp_c,
            altitude_raw_m=altitude_raw,
            accel_body=accel_body,
            gyro_body=gyro_body,
            mag_body=mag_calibrated
        )

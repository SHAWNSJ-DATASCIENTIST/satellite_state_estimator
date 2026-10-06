# 🛰️ Orbital Kalman Filter & State Estimator (Model Satellite ADCS)

A production-grade Flask application providing real-time **Extended Kalman Filtering (EKF)** and **State Estimation** for model satellites, CubeSats, CanSats, and suborbital payloads.

The system fuses four primary sensor subsystems:
1. **BMP280 Barometer**: Atmospheric pressure and temperature (altitude and atmospheric re-entry boundary detection).
2. **MPU6050 6-DOF IMU**: 3-axis accelerometer and 3-axis gyroscope.
3. **LSM6DSM 3-DOF Magnetometer**: Earth geomagnetic field vectors.
4. **Real-Time Clock (RTC)**: Precision UTC timestamps providing dynamic $\Delta t$ time propagation.

The primary mission objectives are to:
- Estimate the full 3D attitude (quaternion $[q_0, q_1, q_2, q_3]$, roll $\phi$, pitch $\theta$, yaw $\psi$).
- Dynamically **calculate the NEXT HEADING** ($\psi_{k+1}$).
- Predict **whether the satellite is still in orbit** or undergoing orbital decay / atmospheric re-entry.

---

## 🌟 Key Features

- **7-State Quaternion Extended Kalman Filter (`AttitudeEKF`)**:
  - State: $\mathbf{x} = [q_0, q_1, q_2, q_3, b_{gx}, b_{gy}, b_{gz}]^T$.
  - Continuous gyro propagation + Discrete Accelerometer tilt correction + Magnetometer yaw correction.
  - Estimated online gyroscope bias tracking to prevent angular drift.
  - Gimbal-lock-free 3D representation.
- **Next Heading Prediction Engine**:
  - Calculates heading angular rate $\dot{\psi} = \frac{\omega_y \sin\phi + \omega_z \cos\phi}{\cos\theta}$.
  - Computes next heading at $t + \Delta t$ and future projected trajectory horizons ($+1\text{s}$, $+2\text{s}$, $+5\text{s}$, $+10\text{s}$).
- **Vertical Kinematics & Orbit Decay Kalman Filter (`AltitudeKalmanFilter`)**:
  - State: $\mathbf{x}_{\text{alt}} = [h, v_z, a_z]^T$ (Altitude, Vertical Velocity, Vertical Acceleration).
  - Rotates MPU6050 body accelerations into the Earth NED frame using the estimated quaternion $\mathbf{R}_{b \to n}(\mathbf{q})$, subtracting gravity to extract true vertical inertial acceleration.
  - Fuses with BMP280 barometric altitude for smooth, zero-latency altitude and descent velocity ($dh/dt$).
- **Orbit Health & De-orbit Predictor (`OrbitPredictor`)**:
  - Classifies flight regime: `STABLE_ORBIT`, `ORBITAL_DECAY`, `CRITICAL_REENTRY`, or `DEORBITED`.
  - Predicts whether satellite is still in orbit (`is_in_orbit: true / false`).
  - Calculates estimated time to re-entry ($T_{\text{reentry}}$), spin stability, tumble index ($\|\vec{\omega}\|$), and aerodynamic drag force.
- **Mission Control Web Dashboard**:
  - Dark-mode telemetry HUD with real-time gauges.
  - Interactive **3D Satellite Model** (Three.js WebGL) reflecting live attitude quaternion.
  - Live Compass Rose & Artificial Horizon (HTML5 2D Canvas).
  - Real-time telemetry charts (Chart.js) for altitude, descent rate, heading, and gyro rates.
  - Built-in simulation controller and custom JSON telemetry injection modal.
- **Realistic Telemetry Simulator & CLI Streamer**:
  - Includes pre-built flight scenarios: `stable_orbit`, `orbital_decay`, `critical_reentry`, and `suborbital_cansat`.

---

## 📐 Mathematical Formulation

### 1. Attitude Propagation (MPU6050 Gyroscope)
Given measured angular velocity $\vec{\omega}_m$ and estimated bias $\mathbf{b}_g$:
$$\hat{\vec{\omega}} = \vec{\omega}_m - \mathbf{b}_g$$
The quaternion kinematics are propagated via:
$$\mathbf{q}_{k} = \left(\mathbf{I}_{4\times 4} + \frac{1}{2} \boldsymbol{\Omega}(\hat{\vec{\omega}}) \Delta t\right) \mathbf{q}_{k-1}$$
where:
$$\boldsymbol{\Omega}(\hat{\vec{\omega}}) = \begin{bmatrix} 0 & -\hat{\omega}_x & -\hat{\omega}_y & -\hat{\omega}_z \\ \hat{\omega}_x & 0 & \hat{\omega}_z & -\hat{\omega}_y \\ \hat{\omega}_y & -\hat{\omega}_z & 0 & \hat{\omega}_x \\ \hat{\omega}_z & \hat{\omega}_y & -\hat{\omega}_x & 0 \end{bmatrix}$$

### 2. Next Heading Calculation
From the estimated Euler angles (Roll $\phi$, Pitch $\theta$, Yaw/Heading $\psi$), the heading derivative is:
$$\dot{\psi} = \frac{\hat{\omega}_y \sin\phi + \hat{\omega}_z \cos\phi}{\cos\theta}$$
The predicted next heading after lookahead horizon $\tau$ is:
$$\psi_{next} = (\psi + \dot{\psi} \cdot \tau) \pmod{360^\circ}$$

### 3. Orbit Determination & Re-entry Criteria
A satellite is evaluated based on four coupled state variables:
1. **Altitude ($h$)**: Above Kármán line ($100\text{ km}$) for space orbit, or above configured floor for suborbital CanSat testing.
2. **Descent Rate ($v_z = dh/dt$)**: Nominal orbital decay exhibits $v_z \approx 0\text{ m/s}$. Rapid descent ($v_z < -25\text{ m/s}$) signals ballistic re-entry.
3. **Barometric Pressure ($P$)**: High vacuum ($P < 0.1\text{ Pa}$) confirms orbital space. Pressure surge indicates dense atmospheric interface entry.
4. **Tumble Index ($\|\vec{\omega}\|$)**: An uncontrolled tumbling satellite ($\|\vec{\omega}\| > 45^\circ/\text{s}$) experiences erratic cross-sectional area changes and massive drag multiplication, accelerating orbital loss.

---

## 📁 Project Architecture

```
satellite_state_estimator/
├── app.py                      # Flask Application entry point & REST API routes
├── config.py                   # Configuration parameters, thresholds & covariances
├── requirements.txt            # Python dependencies
├── simulate_cli.py             # CLI simulation runner & HTTP telemetry streamer
├── satellite/
│   ├── __init__.py
│   ├── sensors.py              # Telemetry parsing & SI unit normalization
│   ├── orbit_predictor.py      # Orbit health, re-entry & stability classification
│   ├── simulator.py            # 6-DOF synthetic flight telemetry generator
│   └── kalman/
│       ├── __init__.py
│       ├── attitude_ekf.py     # 7-State Attitude EKF (Quaternion + Next Heading)
│       ├── altitude_kf.py      # 3-State Altitude & Vertical Velocity Kalman Filter
│       └── fused_estimator.py  # Fused 6-DOF Satellite State Estimator
├── static/
│   ├── css/
│   │   └── dashboard.css       # Space-grade Mission Control HUD styles
│   └── js/
│       └── dashboard.js        # Three.js 3D satellite, Canvas gauges & Chart.js logic
├── templates/
│   └── index.html              # Mission Control Web Dashboard UI
└── tests/
    ├── __init__.py
    ├── run_tests.py            # Standard library test runner
    ├── test_kalman.py          # EKF & KF algorithmic verification tests
    ├── test_orbit.py           # Orbit predictor classification tests
    └── test_api.py             # Flask API integration tests
```

---

## 🔌 Hardware Wiring Guide (Model Satellite / CanSat)

Connect all sensors via I2C bus to your flight computer (e.g. Raspberry Pi, ESP32, Teensy, or STM32):

| Subsystem | Sensor IC | Interface | Default I2C Addr | Pins |
| :--- | :--- | :--- | :--- | :--- |
| **Barometer** | BMP280 | I2C / SPI | `0x76` or `0x77` | `VCC (3.3V)`, `GND`, `SCL`, `SDA` |
| **IMU** | MPU6050 | I2C | `0x68` or `0x69` | `VCC (3.3V/5V)`, `GND`, `SCL`, `SDA`, `INT` |
| **Magnetometer** | LSM6DSM / Mag | I2C / SPI | `0x6A` or `0x6B` | `VCC (3.3V)`, `GND`, `SCL`, `SDA` |
| **RTC** | DS3231 / PCF8563 | I2C | `0x68` or `0x51` | `VCC (3.3V)`, `GND`, `SCL`, `SDA` |

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
cd C:\Users\front\.gemini\antigravity\scratch\satellite_state_estimator
pip install -r requirements.txt
```

### 2. Run the Verification Tests
Execute the test suite to verify filter convergence and API responses:
```bash
python tests/run_tests.py
```

### 3. Launch the Flask Mission Control Server
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### 4. Interactive Dashboard Controls
Once the web dashboard opens:
- Click **"▶ Start Live Stream"** to watch the real-time Kalman filter converge and update the 3D satellite model.
- Switch scenarios (`Stable LEO Orbit`, `Orbital Decay`, `Critical Re-entry`, `Suborbital CanSat`).
- Click **"📡 Inject Custom Packet"** to send your own sensor values directly into the pipeline.

### 5. Run the CLI Telemetry Streamer
To test without opening the browser:
```bash
# Run local offline simulation in terminal
python simulate_cli.py --mode local --scenario stable_orbit --steps 20

# Or stream simulated packets directly into the running Flask API:
python simulate_cli.py --mode stream --scenario orbital_decay --steps 50 --url http://127.0.0.1:5000
```

---

## 📡 REST API Reference

### 1. Ingest Telemetry Packet
**`POST /api/telemetry`**

**Request Body (`application/json`):**
```json
{
  "rtc_time": 1728245000.0,
  "bmp280": {
    "pressure_pa": 0.005,
    "temperature_c": -25.0
  },
  "mpu6050": {
    "accel_x": 0.01,
    "accel_y": -0.02,
    "accel_z": 0.05,
    "gyro_x": 0.005,
    "gyro_y": 0.002,
    "gyro_z": 0.012
  },
  "lsm6dsm": {
    "mag_x": 21.5,
    "mag_y": -4.2,
    "mag_z": 41.8
  }
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "timestamp": 1728245000.0,
  "attitude": {
    "quaternion": [0.9999, 0.001, -0.002, 0.008],
    "roll_deg": 0.12,
    "pitch_deg": -0.23,
    "yaw_deg": 45.2,
    "current_heading_deg": 45.2,
    "next_heading_deg": 46.1,
    "heading_rate_deg_s": 0.9,
    "tumble_rate_deg_s": 1.2
  },
  "altitude_state": {
    "filtered_altitude_m": 300000.0,
    "vertical_velocity_mps": -0.05,
    "vertical_accel_mps2": 0.001,
    "drag_accel_estimate_mps2": 0.0
  },
  "orbit_prediction": {
    "is_in_orbit": true,
    "orbit_status": "STABLE_ORBIT",
    "severity": "nominal",
    "stability_score": 98.5,
    "estimated_reentry_time_sec": null,
    "summary_message": "Nominal stable orbit at 300.0 km altitude. Stable heading drift."
  },
  "next_heading": {
    "current_heading_deg": 45.2,
    "next_heading_deg": 46.1,
    "heading_rate_deg_s": 0.9,
    "trajectory": [
      { "delta_sec": 0.5, "projected_heading_deg": 45.65 },
      { "delta_sec": 1.0, "projected_heading_deg": 46.10 },
      { "delta_sec": 2.0, "projected_heading_deg": 47.00 },
      { "delta_sec": 5.0, "projected_heading_deg": 49.70 },
      { "delta_sec": 10.0, "projected_heading_deg": 54.20 }
    ]
  }
}
```

### 2. Other Endpoints
- `POST /api/telemetry/batch`: Ingest an array of packets for sequential replay.
- `GET /api/state`: Retrieve the most recent estimated state.
- `GET /api/history?limit=100`: Retrieve the last $N$ states for plotting.
- `POST /api/reset`: Reset filter states.
- `POST /api/simulation/start`: Start background 10 Hz simulation streamer.
- `POST /api/simulation/stop`: Stop simulation streamer.
- `GET /api/config`: Read current configuration parameters.
- `POST /api/config`: Update mission mode (`ORBITAL` or `SUBORBITAL_TEST`) or lookahead time.

---

## 🐍 Client Example: Streaming from Satellite Flight Computer

```python
import time
import requests
# from smbus2 import SMBus (or your sensor drivers)

API_ENDPOINT = "http://groundstation-ip:5000/api/telemetry"

while True:
    # 1. Read hardware sensors
    rtc_timestamp = time.time()
    bmp_pressure_pa = 0.015       # Read from BMP280
    bmp_temp_c = -20.0
    ax, ay, az = 0.01, -0.02, 0.05 # Read from MPU6050
    gx, gy, gz = 0.005, 0.002, 0.01
    mx, my, mz = 21.5, -4.2, 41.8  # Read from LSM6DSM

    packet = {
        "rtc_time": rtc_timestamp,
        "bmp280": {"pressure_pa": bmp_pressure_pa, "temperature_c": bmp_temp_c},
        "mpu6050": {
            "accel_x": ax, "accel_y": ay, "accel_z": az,
            "gyro_x": gx, "gyro_y": gy, "gyro_z": gz
        },
        "lsm6dsm": {"mag_x": mx, "mag_y": my, "mag_z": mz}
    }

    response = requests.post(API_ENDPOINT, json=packet)
    data = response.json()
    
    print(f"Next Heading: {data['next_heading']['next_heading_deg']}° | Still in Orbit: {data['orbit_prediction']['is_in_orbit']}")
    time.sleep(0.1)
```

/**
 * Satellite Mission Control & Kalman Filter Dashboard Logic
 * Integrates Three.js 3D CubeSat view, 2D HTML5 Canvas gauges, Chart.js telemetry plots,
 * and REST API communications with the Flask backend.
 */

let altitudeChart, headingChart, gyroChart;
let threeScene, threeCamera, threeRenderer, satelliteGroup;
let isStreaming = false;
let pollInterval = null;

// ==========================================
// 1. INITIALIZATION & LIFECYCLE
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initCharts();
    setupEventListeners();
    drawCompass(0, 0);
    drawHorizon(0, 0);
    
    // Check initial status
    fetchSimulationStatus();
});

function setupEventListeners() {
    // Control bar buttons
    document.getElementById('btn-start-sim').addEventListener('click', startSimulation);
    document.getElementById('btn-stop-sim').addEventListener('click', stopSimulation);
    document.getElementById('btn-step-sim').addEventListener('click', stepSimulation);
    document.getElementById('btn-reset').addEventListener('click', resetFilters);
    
    // Manual modal
    const modal = document.getElementById('injection-modal');
    document.getElementById('btn-toggle-manual').addEventListener('click', () => modal.classList.remove('hidden'));
    document.getElementById('btn-close-modal').addEventListener('click', () => modal.classList.add('hidden'));
    document.getElementById('btn-modal-cancel').addEventListener('click', () => modal.classList.add('hidden'));
    document.getElementById('btn-send-packet').addEventListener('click', sendCustomTelemetry);

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => loadPreset(e.target.dataset.preset));
    });

    // Resize handler for 3D viewport & charts
    window.addEventListener('resize', onWindowResize);
}

// ==========================================
// 2. THREE.JS 3D SATELLITE VISUALIZATION
// ==========================================
function initThreeJS() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x020617);

    threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    threeCamera.position.set(0, 2.8, 4.5);
    threeCamera.lookAt(0, 0, 0);

    threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(threeRenderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    threeScene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight1.position.set(5, 10, 7);
    threeScene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf59e0b, 0.4);
    dirLight2.position.set(-5, -5, -5);
    threeScene.add(dirLight2);

    // Build CubeSat Model
    satelliteGroup = new THREE.Group();

    // Chassis (CubeSat main body)
    const bodyGeo = new THREE.BoxGeometry(1.2, 1.2, 1.8);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.85,
        roughness: 0.25
    });
    const chassis = new THREE.Mesh(bodyGeo, bodyMat);
    satelliteGroup.add(chassis);

    // Gold thermal foil accent
    const foilGeo = new THREE.BoxGeometry(1.22, 0.6, 1.22);
    const foilMat = new THREE.MeshStandardMaterial({
        color: 0xd97706,
        metalness: 0.95,
        roughness: 0.15
    });
    const foil = new THREE.Mesh(foilGeo, foilMat);
    satelliteGroup.add(foil);

    // Solar Panels (Left & Right)
    const panelGeo = new THREE.BoxGeometry(1.8, 0.05, 1.4);
    const panelMat = new THREE.MeshStandardMaterial({
        color: 0x0284c7,
        metalness: 0.5,
        roughness: 0.2
    });

    const leftPanel = new THREE.Mesh(panelGeo, panelMat);
    leftPanel.position.set(1.5, 0, 0);
    satelliteGroup.add(leftPanel);

    const rightPanel = new THREE.Mesh(panelGeo, panelMat);
    rightPanel.position.set(-1.5, 0, 0);
    satelliteGroup.add(rightPanel);

    // Antenna & Nadir payload lens
    const antGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.8, 8);
    const antMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.9 });
    const antenna = new THREE.Mesh(antGeo, antMat);
    antenna.position.set(0, 0, 1.3);
    antenna.rotation.x = Math.PI / 2;
    satelliteGroup.add(antenna);

    // Coordinate Axes Helper
    const axesHelper = new THREE.AxesHelper(1.8);
    satelliteGroup.add(axesHelper);

    threeScene.add(satelliteGroup);

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();
}

function updateSatelliteOrientation(quaternionArr) {
    if (!satelliteGroup || !quaternionArr) return;
    // Quaternion in EKF is [q0, q1, q2, q3] where q0 is scalar w, and (q1, q2, q3) is (x, y, z)
    // Three.js Quaternion constructor is (x, y, z, w)
    const [w, x, y, z] = quaternionArr;
    
    // Convert aerospace NED frame orientation to Three.js coordinate system:
    // In Three.js: +Y is Up, +Z is Forward/Out, +X is Right
    const qThree = new THREE.Quaternion(x, -z, y, w);
    satelliteGroup.quaternion.copy(qThree);
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    if (container && threeCamera && threeRenderer) {
        const width = container.clientWidth;
        const height = container.clientHeight;
        threeCamera.aspect = width / height;
        threeCamera.updateProjectionMatrix();
        threeRenderer.setSize(width, height);
    }
}

// ==========================================
// 3. CANVAS 2D GAUGES (COMPASS & HORIZON)
// ==========================================
function drawCompass(currentHeadingDeg, nextHeadingDeg) {
    const canvas = document.getElementById('compass-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const center = width / 2;
    const radius = center - 12;

    ctx.clearRect(0, 0, width, height);

    // Outer Dial Ring
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Background gradient
    const bgGrad = ctx.createRadialGradient(center, center, 10, center, center, radius);
    bgGrad.addColorStop(0, 'rgba(15, 23, 42, 0.8)');
    bgGrad.addColorStop(1, 'rgba(2, 6, 23, 0.95)');
    ctx.fillStyle = bgGrad;
    ctx.fill();

    // Predicted Next Heading Arc / Wedge
    const curRad = (currentHeadingDeg - 90) * (Math.PI / 180);
    const nextRad = (nextHeadingDeg - 90) * (Math.PI / 180);

    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.arc(center, center, radius - 4, curRad, nextRad, curRad > nextRad);
    ctx.closePath();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.fill();

    // Compass Cardinal Marks
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px "Chakra Petch", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const cardinals = [
        { text: 'N', angle: 0, color: '#ef4444' },
        { text: 'E', angle: 90, color: '#94a3b8' },
        { text: 'S', angle: 180, color: '#94a3b8' },
        { text: 'W', angle: 270, color: '#94a3b8' }
    ];

    cardinals.forEach(c => {
        const rad = (c.angle - 90) * (Math.PI / 180);
        const x = center + (radius - 12) * Math.cos(rad);
        const y = center + (radius - 12) * Math.sin(rad);
        ctx.fillStyle = c.color;
        ctx.fillText(c.text, x, y);
    });

    // Predicted Next Heading Needle (Cyan dashed pointer)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(center + (radius - 6) * Math.cos(nextRad), center + (radius - 6) * Math.sin(nextRad));
    ctx.stroke();
    ctx.setLineDash([]);

    // Current Heading Needle (Solid White/Red arrow)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(center + (radius - 8) * Math.cos(curRad), center + (radius - 8) * Math.sin(curRad));
    ctx.stroke();

    // Center Hub
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(center, center, 4, 0, 2 * Math.PI);
    ctx.fill();
}

function drawHorizon(rollDeg, pitchDeg) {
    const canvas = document.getElementById('horizon-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const center = width / 2;
    const radius = center - 12;

    ctx.clearRect(0, 0, width, height);

    ctx.save();
    // Clip circular boundary
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, 2 * Math.PI);
    ctx.clip();

    // Rotate canvas by -rollDeg
    const rollRad = -rollDeg * (Math.PI / 180);
    const pitchOffset = (pitchDeg / 90) * (radius * 0.8);

    ctx.translate(center, center);
    ctx.rotate(rollRad);

    // Sky half (Top)
    ctx.fillStyle = '#0284c7';
    ctx.fillRect(-radius, -radius * 2 + pitchOffset, radius * 2, radius * 2);

    // Earth half (Bottom)
    ctx.fillStyle = '#78350f';
    ctx.fillRect(-radius, pitchOffset, radius * 2, radius * 2);

    // Horizon line
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-radius, pitchOffset);
    ctx.lineTo(radius, pitchOffset);
    ctx.stroke();

    ctx.restore();

    // Static aircraft reticle (Yellow target in center)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(center - 24, center);
    ctx.lineTo(center - 8, center);
    ctx.lineTo(center - 8, center + 6);
    ctx.moveTo(center + 24, center);
    ctx.lineTo(center + 8, center);
    ctx.lineTo(center + 8, center + 6);
    ctx.stroke();

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(center, center, 3, 0, 2 * Math.PI);
    ctx.fill();

    // Outer bezel ring
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, 2 * Math.PI);
    ctx.stroke();
}

// ==========================================
// 4. CHART.JS REAL-TIME TELEMETRY PLOTS
// ==========================================
function initCharts() {
    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
            }
        },
        scales: {
            x: {
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 }, maxTicksLimit: 6 },
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            y: {
                ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 9 } },
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
        }
    };

    // Chart 1: Altitude & Descent Velocity
    const ctxAlt = document.getElementById('altitudeChart').getContext('2d');
    altitudeChart = new Chart(ctxAlt, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Filtered Altitude (km)',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y',
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'Vertical Velocity (m/s)',
                    data: [],
                    borderColor: '#f43f5e',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    yAxisID: 'y1',
                    tension: 0.2
                }
            ]
        },
        options: {
            ...commonChartOptions,
            scales: {
                ...commonChartOptions.scales,
                y1: {
                    position: 'right',
                    ticks: { color: '#f43f5e', font: { family: 'JetBrains Mono', size: 9 } },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });

    // Chart 2: Current Heading vs Predicted Next Heading
    const ctxHead = document.getElementById('headingChart').getContext('2d');
    headingChart = new Chart(ctxHead, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Current Heading (°)',
                    data: [],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Predicted Next Heading (°)',
                    data: [],
                    borderColor: '#06b6d4',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: commonChartOptions
    });

    // Chart 3: Gyro Rates
    const ctxGyro = document.getElementById('gyroChart').getContext('2d');
    gyroChart = new Chart(ctxGyro, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'ωx (Roll rate)', data: [], borderColor: '#ef4444', borderWidth: 1.5 },
                { label: 'ωy (Pitch rate)', data: [], borderColor: '#22c55e', borderWidth: 1.5 },
                { label: 'ωz (Yaw rate)', data: [], borderColor: '#3b82f6', borderWidth: 1.5 }
            ]
        },
        options: commonChartOptions
    });
}

function updateCharts(history) {
    if (!history || history.length === 0) return;
    const maxPoints = 30;
    const slice = history.slice(-maxPoints);

    const labels = slice.map((item, idx) => {
        const t = new Date(item.timestamp * 1000);
        return t.toTimeString().split(' ')[0];
    });

    // Update Altitude & Velocity
    altitudeChart.data.labels = labels;
    altitudeChart.data.datasets[0].data = slice.map(item => item.altitude_state.filtered_altitude_m / 1000.0);
    altitudeChart.data.datasets[1].data = slice.map(item => item.altitude_state.vertical_velocity_mps);
    altitudeChart.update('none');

    // Update Headings
    headingChart.data.labels = labels;
    headingChart.data.datasets[0].data = slice.map(item => item.attitude.current_heading_deg);
    headingChart.data.datasets[1].data = slice.map(item => item.attitude.next_heading_deg);
    headingChart.update('none');

    // Update Gyro
    gyroChart.data.labels = labels;
    gyroChart.data.datasets[0].data = slice.map(item => item.raw_sensors.mpu6050.gyro_x_deg_s);
    gyroChart.data.datasets[1].data = slice.map(item => item.raw_sensors.mpu6050.gyro_y_deg_s);
    gyroChart.data.datasets[2].data = slice.map(item => item.raw_sensors.mpu6050.gyro_z_deg_s);
    gyroChart.update('none');
}

// ==========================================
// 5. TELEMETRY STATE & HUD UPDATES
// ==========================================
function updateDashboardUI(payload) {
    const att = payload.attitude;
    const alt = payload.altitude_state;
    const orbit = payload.orbit_prediction;
    const sensors = payload.raw_sensors;

    // 1. Heading & Prediction
    document.getElementById('current-heading-val').textContent = att.current_heading_deg.toFixed(2);
    document.getElementById('next-heading-val').textContent = att.next_heading_deg.toFixed(2);
    document.getElementById('heading-cardinal').textContent = getCardinalDirection(att.current_heading_deg);
    document.getElementById('next-heading-delta').textContent = `Rate: ${att.heading_rate_deg_s > 0 ? '+' : ''}${att.heading_rate_deg_s.toFixed(2)}°/s`;

    // Euler breakdown
    document.getElementById('roll-val').textContent = `${att.roll_deg.toFixed(2)}°`;
    document.getElementById('pitch-val').textContent = `${att.pitch_deg.toFixed(2)}°`;
    document.getElementById('yaw-val').textContent = `${att.yaw_deg.toFixed(2)}°`;
    document.getElementById('yaw-rate-val').textContent = `${att.heading_rate_deg_s.toFixed(2)}°/s`;
    document.getElementById('tumble-rate-val').textContent = `${att.tumble_rate_deg_s.toFixed(2)}°/s`;

    // 2D Gauges
    drawCompass(att.current_heading_deg, att.next_heading_deg);
    drawHorizon(att.roll_deg, att.pitch_deg);

    // 3D Model
    updateSatelliteOrientation(att.quaternion);

    // 2. Altitude & Kinematics
    const altKm = (alt.filtered_altitude_m / 1000.0).toFixed(1);
    const rawAltKm = (alt.raw_bmp_altitude_m / 1000.0).toFixed(1);
    document.getElementById('altitude-val').textContent = altKm;
    document.getElementById('bmp-raw-alt').textContent = `BMP280 Raw: ${rawAltKm} km`;
    document.getElementById('decay-rate-val').textContent = alt.vertical_velocity_mps.toFixed(2);

    const decayText = document.getElementById('decay-status-text');
    if (alt.vertical_velocity_mps < -2.0) {
        decayText.textContent = `Rapid Decay (${Math.abs(alt.vertical_velocity_mps).toFixed(1)} m/s)`;
        decayText.style.color = '#ef4444';
    } else if (alt.vertical_velocity_mps < -0.1) {
        decayText.textContent = 'Nominal Decay';
        decayText.style.color = '#f59e0b';
    } else {
        decayText.textContent = 'Stable / Zero Descent';
        decayText.style.color = '#10b981';
    }

    // 3. Orbit Status & Health
    updateOrbitStatusBadge(orbit.orbit_status, orbit.severity);
    document.getElementById('stability-health-val').textContent = `${orbit.stability_score}%`;
    document.getElementById('stability-health-fill').style.width = `${orbit.stability_score}%`;

    const healthFill = document.getElementById('stability-health-fill');
    if (orbit.stability_score > 70) healthFill.style.background = '#10b981';
    else if (orbit.stability_score > 35) healthFill.style.background = '#f59e0b';
    else healthFill.style.background = '#ef4444';

    // Diagnostic Items
    const orbitTag = document.getElementById('is-in-orbit-tag');
    orbitTag.textContent = orbit.is_in_orbit ? 'YES (IN ORBIT)' : 'NO (DE-ORBITED)';
    orbitTag.className = orbit.is_in_orbit ? 'diag-val tag-true' : 'diag-val tag-false';

    document.getElementById('reentry-time-val').textContent = orbit.estimated_reentry_time_sec 
        ? `${orbit.estimated_reentry_time_sec.toFixed(0)} sec (${(orbit.estimated_reentry_time_sec / 60).toFixed(1)} min)` 
        : 'STABLE (N/A)';

    document.getElementById('spin-stability-val').textContent = orbit.spin_stability;
    document.getElementById('drag-val').textContent = `${alt.drag_accel_estimate_mps2} m/s²`;
    document.getElementById('density-val').textContent = `${alt.air_density_kg_m3} kg/m³`;

    // Projected Heading Horizons chips
    if (att.trajectory && att.trajectory.length >= 4) {
        const chipsContainer = document.getElementById('heading-chips');
        chipsContainer.innerHTML = att.trajectory.slice(0, 4).map(h => 
            `<div class="chip">+${h.delta_sec}s: <span class="mono">${h.projected_heading_deg.toFixed(1)}°</span></div>`
        ).join('');
    }

    // 4. Raw Sensor Subsystems
    document.getElementById('p-val').textContent = sensors.bmp280.pressure_pa;
    document.getElementById('t-val').textContent = sensors.bmp280.temperature_c;
    document.getElementById('accel-xyz').textContent = 
        `X:${sensors.mpu6050.accel_x.toFixed(2)} Y:${sensors.mpu6050.accel_y.toFixed(2)} Z:${sensors.mpu6050.accel_z.toFixed(2)}`;
    document.getElementById('gyro-xyz').textContent = 
        `X:${sensors.mpu6050.gyro_x_deg_s.toFixed(1)} Y:${sensors.mpu6050.gyro_y_deg_s.toFixed(1)} Z:${sensors.mpu6050.gyro_z_deg_s.toFixed(1)}`;
    document.getElementById('mag-xyz').textContent = 
        `X:${sensors.lsm6dsm.mag_x_uT.toFixed(1)} Y:${sensors.lsm6dsm.mag_y_uT.toFixed(1)} Z:${sensors.lsm6dsm.mag_z_uT.toFixed(1)}`;

    // RTC Clock
    const rtcDate = new Date(payload.timestamp * 1000);
    document.getElementById('rtc-time-display').textContent = rtcDate.toTimeString().split(' ')[0] + ' UTC';

    // Banner message
    const banner = document.getElementById('status-banner');
    document.getElementById('banner-text').textContent = orbit.summary_message;
    if (orbit.severity === 'nominal') banner.className = 'status-banner banner-info';
    else if (orbit.severity === 'warning') banner.className = 'status-banner banner-warning';
    else banner.className = 'status-banner banner-critical';
}

function updateOrbitStatusBadge(status, severity) {
    const badge = document.getElementById('orbit-status-badge');
    badge.textContent = status.replace('_', ' ');

    if (status === 'STABLE_ORBIT') badge.className = 'badge-status status-stable';
    else if (status === 'ORBITAL_DECAY') badge.className = 'badge-status status-decay';
    else if (status === 'CRITICAL_REENTRY') badge.className = 'badge-status status-reentry';
    else badge.className = 'badge-status status-deorbited';
}

function getCardinalDirection(deg) {
    const val = Math.floor((deg / 22.5) + 0.5);
    const arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
    return arr[(val % 16)];
}

// ==========================================
// 6. BACKEND API CALLS & STREAMING
// ==========================================
async function fetchLatestStateAndHistory() {
    try {
        const histRes = await fetch('/api/history?limit=30');
        const histData = await histRes.json();
        if (histData.success && histData.history.length > 0) {
            const latest = histData.history[histData.history.length - 1];
            updateDashboardUI(latest);
            updateCharts(histData.history);
        }
    } catch (err) {
        console.error("Telemetry fetch error:", err);
    }
}

async function fetchSimulationStatus() {
    try {
        const res = await fetch('/api/simulation/status');
        const data = await res.json();
        setStreamRunningUI(data.is_running);
        if (data.is_running) {
            startPolling();
        }
    } catch (err) {
        console.error("Status check error:", err);
    }
}

async function startSimulation() {
    const scenario = document.getElementById('scenario-select').value;
    try {
        const res = await fetch('/api/simulation/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario })
        });
        const data = await res.json();
        if (data.success) {
            setStreamRunningUI(true);
            startPolling();
        }
    } catch (err) {
        console.error("Failed to start simulation:", err);
    }
}

async function stopSimulation() {
    try {
        const res = await fetch('/api/simulation/stop', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            setStreamRunningUI(false);
            stopPolling();
        }
    } catch (err) {
        console.error("Failed to stop simulation:", err);
    }
}

async function stepSimulation() {
    const scenario = document.getElementById('scenario-select').value;
    try {
        const res = await fetch('/api/simulation/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario })
        });
        const payload = await res.json();
        if (payload.success) {
            updateDashboardUI(payload);
            fetchLatestStateAndHistory();
        }
    } catch (err) {
        console.error("Step simulation error:", err);
    }
}

async function resetFilters() {
    try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            fetchLatestStateAndHistory();
            drawCompass(0, 0);
            drawHorizon(0, 0);
        }
    } catch (err) {
        console.error("Reset error:", err);
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchLatestStateAndHistory, 250);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function setStreamRunningUI(running) {
    isStreaming = running;
    document.getElementById('btn-start-sim').disabled = running;
    document.getElementById('btn-stop-sim').disabled = !running;
    
    const pill = document.getElementById('sim-status-indicator');
    if (running) {
        pill.textContent = 'STREAMING (10Hz)';
        pill.className = 'sim-pill on';
    } else {
        pill.textContent = 'OFFLINE';
        pill.className = 'sim-pill off';
    }
}

// ==========================================
// 7. CUSTOM TELEMETRY PACKET INJECTION
// ==========================================
async function sendCustomTelemetry() {
    const text = document.getElementById('telemetry-json-input').value;
    try {
        const json = JSON.parse(text);
        const res = await fetch('/api/telemetry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(json)
        });
        const result = await res.json();
        if (result.success) {
            document.getElementById('injection-modal').classList.add('hidden');
            updateDashboardUI(result);
            fetchLatestStateAndHistory();
        } else {
            alert('Error from Kalman Filter: ' + result.error);
        }
    } catch (e) {
        alert('Invalid JSON: ' + e.message);
    }
}

function loadPreset(preset) {
    const editor = document.getElementById('telemetry-json-input');
    const now = Math.floor(Date.now() / 1000);

    if (preset === 'leo') {
        editor.value = JSON.stringify({
            rtc_time: now,
            bmp280: { pressure_pa: 0.005, temperature_c: -35.0 },
            mpu6050: { accel_x: 0.01, accel_y: 0.02, accel_z: 0.01, gyro_x: 0.005, gyro_y: 0.002, gyro_z: 0.015 },
            lsm6dsm: { mag_x: 21.0, mag_y: -4.0, mag_z: 42.0 }
        }, null, 2);
    } else if (preset === 'decay') {
        editor.value = JSON.stringify({
            rtc_time: now,
            bmp280: { pressure_pa: 0.8, temperature_c: -45.0 },
            mpu6050: { accel_x: 0.05, accel_y: -0.04, accel_z: -0.15, gyro_x: 0.08, gyro_y: -0.05, gyro_z: 0.12 },
            lsm6dsm: { mag_x: 18.0, mag_y: 12.0, mag_z: 38.0 }
        }, null, 2);
    } else if (preset === 'reentry') {
        editor.value = JSON.stringify({
            rtc_time: now,
            bmp280: { pressure_pa: 150.0, temperature_c: 120.0 },
            mpu6050: { accel_x: 1.5, accel_y: -2.3, accel_z: -9.8, gyro_x: 1.2, gyro_y: -1.8, gyro_z: 2.5 },
            lsm6dsm: { mag_x: 12.0, mag_y: 25.0, mag_z: 15.0 }
        }, null, 2);
    } else if (preset === 'cansat') {
        editor.value = JSON.stringify({
            rtc_time: now,
            bmp280: { pressure_pa: 75200.0, temperature_c: 8.5 },
            mpu6050: { accel_x: 0.2, accel_y: 0.1, accel_z: 9.81, gyro_x: 0.02, gyro_y: 0.01, gyro_z: 0.05 },
            lsm6dsm: { mag_x: 22.0, mag_y: 5.0, mag_z: 40.0 }
        }, null, 2);
    }
}

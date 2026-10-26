// Industrial Safety Intelligence AI Single-Page Cockpit Controller

let audioEnabled = true;
let isContinuousRunning = false;
let continuousInterval = null;
let currentArchivePage = 1;
let currentArchiveTotalPages = 1;
let activeIncidentsList = [];
let audioCtx = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) lucide.createIcons();
  initClock();
  checkBackendHealth();
  initLiveStation();
  loadZones();
  loadIncidents();
  loadArchiveImages(1);
  setupEventListeners();
});

// Real-Time System Clock
function initClock() {
  const clockEl = document.getElementById('systemClock');
  const update = () => {
    const d = new Date();
    clockEl.innerText = d.toTimeString().split(' ')[0] + ' UTC';
  };
  update();
  setInterval(update, 1000);
}

// Check Backend Health & Models
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = document.getElementById('backendStatusDot');
    const txt = document.getElementById('backendStatusText');
    if (data.status === 'HEALTHY') {
      dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse';
      txt.innerText = 'AI Models Online';
      txt.className = 'text-emerald-400 font-semibold';
    } else {
      dot.className = 'w-2.5 h-2.5 rounded-full bg-amber-500';
      txt.innerText = 'Initializing';
      txt.className = 'text-amber-400 font-semibold';
    }
  } catch (err) {
    const dot = document.getElementById('backendStatusDot');
    const txt = document.getElementById('backendStatusText');
    dot.className = 'w-2.5 h-2.5 rounded-full bg-red-500';
    txt.innerText = 'Backend Offline';
    txt.className = 'text-red-400 font-semibold';
  }
}

// Web Audio API Synthesizer Alarm Siren
function triggerAlarmSound() {
  if (!audioEnabled) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.setValueAtTime(1100, audioCtx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.4);
  } catch (e) {
    console.warn("Audio trigger blocked:", e);
  }
}

// ============================================================================
// 1. PRIMARY CCTV INSPECTION STATION
// ============================================================================
function initLiveStation() {
  const video = document.getElementById('cctvVideo');
  const canvas = document.getElementById('cctvCanvas');
  const ctx = canvas.getContext('2d');

  video.addEventListener('timeupdate', () => {
    const cur = formatTime(video.currentTime);
    const dur = formatTime(video.duration || 0);
    document.getElementById('videoTimeDisplay').innerText = `${cur} / ${dur}`;
  });

  video.addEventListener('loadeddata', () => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 360;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  });
}

async function captureAndInspectFrame() {
  const video = document.getElementById('cctvVideo');
  const canvas = document.getElementById('cctvCanvas');
  const ctx = canvas.getContext('2d');
  const camSelect = document.getElementById('cameraProfileSelect').value;
  const spinner = document.getElementById('videoLoadingSpinner');

  if (video.videoWidth > 0 && video.videoHeight > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }

  spinner.classList.remove('hidden');

  try {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    const formData = new FormData();
    formData.append('file', blob, 'cctv_frame.jpg');

    const res = await fetch(`/api/inspect/image?camera_id=${camSelect}`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    spinner.classList.add('hidden');
    updateLiveTelemetry(data);
  } catch (err) {
    spinner.classList.add('hidden');
    console.error("Frame inspection failed:", err);
  }
}

function updateLiveTelemetry(data) {
  const canvas = document.getElementById('cctvCanvas');
  const ctx = canvas.getContext('2d');

  // 1. Draw annotated frame with HUD
  if (data.annotated_image) {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = data.annotated_image;
  }

  // 2. Latency & FPS
  document.getElementById('hudLatencyMs').innerText = `${data.latency_ms.toFixed(1)} ms`;
  const fps = data.latency_ms > 0 ? (1000 / data.latency_ms).toFixed(1) : '30.0';
  document.getElementById('hudFpsVal').innerText = fps;

  // 3. Safety Score Radial Gauge
  const score = Math.round(data.safety_score);
  document.getElementById('safetyScoreVal').innerText = `${score}%`;
  const circle = document.getElementById('scoreGaugeCircle');
  const dashoffset = 251.2 - (251.2 * score / 100);
  circle.style.strokeDashoffset = dashoffset;
  circle.style.stroke = data.risk_color || '#10b981';

  // 4. Risk Level Badge
  const badge = document.getElementById('riskLevelBadge');
  badge.innerText = data.risk_level;
  const container = document.getElementById('videoContainer');

  if (data.risk_level === 'CRITICAL') {
    badge.className = 'px-3 py-1 rounded-md text-xs font-extrabold tracking-wide uppercase bg-red-950 text-red-400 border border-red-700 glow-red animate-pulse';
    container.classList.add('pulse-critical');
    triggerAlarmSound();
  } else if (data.risk_level === 'HIGH') {
    badge.className = 'px-3 py-1 rounded-md text-xs font-extrabold tracking-wide uppercase bg-amber-950 text-amber-400 border border-amber-700';
    container.classList.remove('pulse-critical');
  } else if (data.risk_level === 'MEDIUM') {
    badge.className = 'px-3 py-1 rounded-md text-xs font-extrabold tracking-wide uppercase bg-yellow-950 text-yellow-400 border border-yellow-700';
    container.classList.remove('pulse-critical');
  } else {
    badge.className = 'px-3 py-1 rounded-md text-xs font-extrabold tracking-wide uppercase bg-emerald-950 text-emerald-400 border border-emerald-700';
    container.classList.remove('pulse-critical');
  }

  // 5. Counters
  document.getElementById('kpiWorkersCount').innerText = data.workers ? data.workers.length : 0;
  document.getElementById('kpiMachineryCount').innerText = data.machinery ? data.machinery.length : 0;
  const violCount = data.explainability_factors ? data.explainability_factors.length : 0;
  document.getElementById('kpiViolationsCount').innerText = violCount;

  // 6. XAI Assessment
  document.getElementById('xaiPrimaryHazard').innerText = data.primary_hazard || 'None - Safe Working Conditions';
  const factorsEl = document.getElementById('xaiFactorsList');
  if (data.explainability_factors && data.explainability_factors.length > 0) {
    factorsEl.innerHTML = data.explainability_factors.map(f =>
      `<div class="p-2 bg-[#161f33] border-l-2 border-red-500 rounded text-slate-300 font-mono text-xs">${f}</div>`
    ).join('');
  } else {
    factorsEl.innerHTML = '<p class="text-slate-500 italic">No hazard factors recorded in current frame.</p>';
  }

  // 7. Prescriptive Actions
  const actionsEl = document.getElementById('xaiActionsList');
  if (data.prescriptive_actions && data.prescriptive_actions.length > 0) {
    actionsEl.innerHTML = data.prescriptive_actions.map(a => `<div>⚡ ${a}</div>`).join('');
  } else {
    actionsEl.innerText = 'Routine automated CCTV surveillance active.';
  }

  // 8. Workers PPE Breakdown
  const wContainer = document.getElementById('workerListContainer');
  if (data.workers && data.workers.length > 0) {
    wContainer.innerHTML = data.workers.map(w => {
      const hTag = w.has_helmet ? '<span class="text-emerald-400">Hardhat: OK</span>' : '<span class="text-red-400">NO HARDHAT</span>';
      const vTag = w.has_vest ? '<span class="text-emerald-400">Vest: OK</span>' : '<span class="text-red-400">NO VEST</span>';
      return `
        <div class="p-2 bg-[#161f33] border border-[#232f48] rounded flex items-center justify-between text-xs">
          <span class="font-bold text-white">Worker #${w.worker_id}</span>
          <div class="flex gap-2 text-[10px]">
            ${hTag} &bull; ${vTag}
          </div>
        </div>
      `;
    }).join('');
  } else {
    wContainer.innerHTML = '<p class="text-slate-500 italic">No personnel detected in active zones.</p>';
  }

  // Refresh incidents list if an incident was generated
  if (data.incident_id) {
    loadIncidents();
  }
}

// ============================================================================
// 2. ARCHIVE DATASET EXPLORER & BATCH AUDIT (2,102 IMAGES)
// ============================================================================
async function loadArchiveImages(page = 1) {
  const search = document.getElementById('archiveSearchInput').value;
  const grid = document.getElementById('archiveImageGrid');
  grid.innerHTML = '<div class="col-span-full py-8 text-center text-slate-400 font-mono text-xs">Loading archive images from c:\\Users\\Dell\\Downloads\\archive\\images...</div>';

  try {
    const url = `/api/archive/images?page=${page}&page_size=12${search ? '&query=' + encodeURIComponent(search) : ''}`;
    const res = await fetch(url);
    const data = await res.json();

    currentArchivePage = data.page;
    currentArchiveTotalPages = data.total_pages;
    document.getElementById('archivePageInfo').innerText = `Page ${data.page} of ${data.total_pages} (${data.total.toLocaleString()} total images)`;

    if (data.images.length === 0) {
      grid.innerHTML = '<div class="col-span-full py-8 text-center text-slate-500 font-mono text-xs">No matching images found in archive.</div>';
      return;
    }

    grid.innerHTML = data.images.map((filename, i) => `
      <div class="group relative bg-[#0e1422] border border-[#232f48] hover:border-cyan-500 rounded-xl overflow-hidden shadow-lg transition flex flex-col">
        <div class="aspect-square bg-black overflow-hidden relative">
          <img src="/api/archive/image/${filename}" loading="lazy" class="w-full h-full object-cover group-hover:scale-105 transition duration-300" alt="${filename}">
          <button onclick="inspectArchiveImage('${filename}')" class="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 flex items-center justify-center text-xs font-semibold text-white gap-1.5 transition">
            <i data-lucide="scan" class="w-4 h-4 text-cyan-400"></i> Deep Inspect
          </button>
        </div>
        <div class="p-2 text-[10px] font-mono text-slate-400 truncate border-t border-[#1e293b]" title="${filename}">
          ${filename}
        </div>
      </div>
    `).join('');

    if (window.lucide) lucide.createIcons();
  } catch (err) {
    grid.innerHTML = `<div class="col-span-full py-8 text-center text-red-400 font-mono text-xs">Failed to load archive: ${err.message}</div>`;
  }
}

// Deep Inspect an Archive Image (loads directly into the CCTV monitor viewport!)
async function inspectArchiveImage(filename) {
  // Scroll up to monitor section smoothly
  document.getElementById('section-monitor').scrollIntoView({ behavior: 'smooth' });

  const spinner = document.getElementById('videoLoadingSpinner');
  spinner.classList.remove('hidden');

  try {
    const camSelect = document.getElementById('cameraProfileSelect').value;
    const res = await fetch('/api/inspect/path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: `c:\\Users\\Dell\\Downloads\\archive\\images\\${filename}`,
        camera_id: camSelect,
      }),
    });
    const data = await res.json();
    spinner.classList.add('hidden');
    updateLiveTelemetry(data);
  } catch (err) {
    spinner.classList.add('hidden');
    alert(`Inspection failed: ${err.message}`);
  }
}

// Automated Batch Safety Audit
async function runBatchSafetyAudit() {
  const container = document.getElementById('batchAuditContainer');
  const progressBar = document.getElementById('batchProgressBar');
  const progressLabel = document.getElementById('batchProgressLabel');
  const resultsSummary = document.getElementById('batchResultsSummary');

  container.classList.remove('hidden');
  resultsSummary.classList.add('hidden');
  progressBar.style.width = '30%';
  progressLabel.innerText = 'Scanning 15 archive images through 6-stage neural pipeline...';

  try {
    const res = await fetch('/api/batch_audit?sample_size=15&camera_id=CAM-01', { method: 'POST' });
    const data = await res.json();

    progressBar.style.width = '100%';
    progressLabel.innerText = `Completed audit on ${data.total_scanned} industrial images!`;

    document.getElementById('batchAvgScore').innerText = `${data.avg_safety_score}%`;
    document.getElementById('batchCriticalCount').innerText = data.critical_count;
    document.getElementById('batchHighCount').innerText = data.high_count;
    document.getElementById('batchLowCount').innerText = data.low_count;

    resultsSummary.classList.remove('hidden');
    loadIncidents(); // reload newly created incidents
  } catch (err) {
    alert(`Batch audit error: ${err.message}`);
  }
}

// ============================================================================
// 3. ZONES & GEOFENCING CONFIGURATION
// ============================================================================
async function loadZones() {
  const container = document.getElementById('zonesCardsContainer');
  try {
    const res = await fetch('/api/zones');
    const data = await res.json();
    const cameras = data.cameras || {};

    container.innerHTML = Object.entries(cameras).map(([camId, cam]) => `
      <div class="bg-[#0e1422] border border-[#232f48] rounded-xl p-5 space-y-4 shadow-lg">
        <div class="flex items-center justify-between border-b border-[#232f48] pb-3">
          <div>
            <h3 class="font-bold text-white text-sm">${cam.camera_name}</h3>
            <span class="text-xs font-mono text-cyan-400">${camId} &bull; ${cam.location}</span>
          </div>
          <span class="text-xs px-2.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">${cam.zones.length} Zones</span>
        </div>

        <div class="space-y-2.5">
          ${cam.zones.map(z => `
            <div class="p-3 bg-[#161f33] border-l-4 rounded" style="border-left-color: ${z.color}">
              <div class="flex justify-between items-center text-xs">
                <span class="font-bold text-white">[${z.type}] ${z.name}</span>
                <span class="font-mono text-[10px] text-slate-400">${z.points.length} Pts</span>
              </div>
              <p class="text-[11px] text-slate-400 mt-1">${z.description}</p>
              <div class="flex gap-2 mt-2 text-[10px] font-mono">
                <span class="text-slate-300 font-semibold">Required PPE:</span>
                <span class="text-cyan-400">${z.required_ppe.join(', ') || 'Standard PPE'}</span>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<p class="text-red-400">Failed to load zones: ${err.message}</p>`;
  }
}

// ============================================================================
// 4. INCIDENT TICKETS & REAL-TIME RESOLUTION WORKFLOW
// ============================================================================
async function loadIncidents() {
  const tbody = document.getElementById('incidentsTableBody');
  const headerBadge = document.getElementById('headerIncidentCount');
  const statusFilter = document.getElementById('incidentFilterStatus').value;

  try {
    let url = '/api/incidents';
    if (statusFilter !== 'ALL') url += `?status=${statusFilter}`;
    const res = await fetch(url);
    const data = await res.json();
    activeIncidentsList = data || [];

    const openCount = activeIncidentsList.filter(i => i.status === 'OPEN').length;
    headerBadge.innerText = openCount;

    if (activeIncidentsList.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="p-4 text-center text-slate-500 italic">No incident tickets matching criteria.</td></tr>';
      return;
    }

    tbody.innerHTML = activeIncidentsList.map(inc => {
      let riskColor = 'bg-emerald-950 text-emerald-400 border-emerald-800';
      if (inc.risk_level === 'CRITICAL') riskColor = 'bg-red-950 text-red-400 border-red-800';
      else if (inc.risk_level === 'HIGH') riskColor = 'bg-amber-950 text-amber-400 border-amber-800';
      else if (inc.risk_level === 'MEDIUM') riskColor = 'bg-yellow-950 text-yellow-400 border-yellow-800';

      return `
        <tr class="hover:bg-[#161f33] transition">
          <td class="p-3.5 font-bold text-white">${inc.incident_id}</td>
          <td class="p-3.5 text-slate-400">${inc.formatted_time}</td>
          <td class="p-3.5 text-cyan-400">${inc.camera_id}</td>
          <td class="p-3.5">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${riskColor}">${inc.risk_level}</span>
          </td>
          <td class="p-3.5 text-slate-200">${inc.primary_hazard}</td>
          <td class="p-3.5">
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${inc.status === 'OPEN' ? 'bg-red-900/40 text-red-300 border border-red-800' : 'bg-emerald-900/40 text-emerald-300 border border-emerald-800'}">${inc.status}</span>
          </td>
          <td class="p-3.5">
            <button onclick="openIncidentModal('${inc.incident_id}')" class="px-3 py-1 bg-[#1a233a] hover:bg-cyan-600 hover:text-white text-cyan-400 rounded transition font-semibold text-xs">
              Inspect Evidence
            </button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-red-400">Failed to load incidents: ${err.message}</td></tr>`;
  }
}

function openIncidentModal(incId) {
  const inc = activeIncidentsList.find(i => i.incident_id === incId);
  if (!inc) return;

  document.getElementById('modalIncidentId').innerText = inc.incident_id;
  document.getElementById('modalCameraId').innerText = inc.camera_id;
  document.getElementById('modalTimestamp').innerText = inc.formatted_time;
  document.getElementById('modalPrimaryHazard').innerText = inc.primary_hazard;
  document.getElementById('modalAction').innerText = inc.recommended_action;
  document.getElementById('modalRiskBadge').innerText = inc.risk_level;

  const imgEl = document.getElementById('modalEvidenceImage');
  if (inc.evidence_image_path) {
    const filename = inc.evidence_image_path.replace(/\\/g, '/').split('/').pop();
    imgEl.src = `/api/evidence/${filename}`;
  } else {
    imgEl.src = '';
  }

  document.getElementById('modalBtnAcknowledge').onclick = async () => {
    await fetch(`/api/incidents/${inc.incident_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'ACKNOWLEDGE', responder_name: 'Safety Control Room' }),
    });
    closeIncidentModal();
    loadIncidents();
  };

  document.getElementById('modalBtnResolve').onclick = async () => {
    const notes = document.getElementById('modalResolutionNotes').value || 'Hazard resolved and verified on floor.';
    await fetch(`/api/incidents/${inc.incident_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'RESOLVE', notes: notes, responder_name: 'Safety Control Room' }),
    });
    closeIncidentModal();
    loadIncidents();
  };

  document.getElementById('incidentModal').classList.remove('hidden');
  if (window.lucide) lucide.createIcons();
}

function closeIncidentModal() {
  document.getElementById('incidentModal').classList.add('hidden');
}

// Event Listeners Setup
function setupEventListeners() {
  // Audio toggle
  document.getElementById('toggleAudioBtn').addEventListener('click', () => {
    audioEnabled = !audioEnabled;
    document.getElementById('audioLabel').innerText = `Alarm: ${audioEnabled ? 'ON' : 'OFF'}`;
    document.getElementById('audioIcon').className = audioEnabled ? 'w-4 h-4 text-cyan-400' : 'w-4 h-4 text-slate-500';
  });

  // Video playback
  const video = document.getElementById('cctvVideo');
  document.getElementById('btnPlayPauseVideo').addEventListener('click', () => {
    if (video.paused) {
      video.play();
      document.getElementById('playPauseIcon').setAttribute('data-lucide', 'pause');
    } else {
      video.pause();
      document.getElementById('playPauseIcon').setAttribute('data-lucide', 'play');
    }
    if (window.lucide) lucide.createIcons();
  });

  // Step Frame
  document.getElementById('btnStepFrame').addEventListener('click', () => {
    captureAndInspectFrame();
  });

  // Run continuous inspection
  document.getElementById('btnRunContinuous').addEventListener('click', () => {
    isContinuousRunning = !isContinuousRunning;
    const btn = document.getElementById('btnRunContinuous');
    if (isContinuousRunning) {
      btn.innerHTML = '<i data-lucide="square" class="w-4 h-4"></i> Stop Continuous';
      btn.className = 'px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-semibold flex items-center gap-2 shadow-lg transition';
      video.play();
      continuousInterval = setInterval(captureAndInspectFrame, 1200);
    } else {
      btn.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i> Run Continuous Inspection';
      btn.className = 'px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold flex items-center gap-2 shadow-lg shadow-cyan-900/30 transition';
      video.pause();
      clearInterval(continuousInterval);
    }
    if (window.lucide) lucide.createIcons();
  });

  // Archive search
  document.getElementById('archiveSearchInput').addEventListener('input', () => {
    loadArchiveImages(1);
  });

  // Archive pagination
  document.getElementById('btnPrevPage').addEventListener('click', () => {
    if (currentArchivePage > 1) loadArchiveImages(currentArchivePage - 1);
  });
  document.getElementById('btnNextPage').addEventListener('click', () => {
    if (currentArchivePage < currentArchiveTotalPages) loadArchiveImages(currentArchivePage + 1);
  });

  // Batch audit trigger
  document.getElementById('btnStartBatchAudit').addEventListener('click', runBatchSafetyAudit);

  // Incidents filter
  document.getElementById('incidentFilterStatus').addEventListener('change', loadIncidents);
  document.getElementById('btnRefreshIncidents').addEventListener('click', loadIncidents);

  // File Upload
  document.getElementById('liveSourceSelect').addEventListener('change', (e) => {
    if (e.target.value === 'upload') {
      document.getElementById('fileUploadInput').click();
    } else if (e.target.value === 'archive_sample') {
      // Pick random archive image
      fetch('/api/archive/images?page=1&page_size=20').then(r => r.json()).then(data => {
        if (data.images && data.images.length > 0) {
          const randImg = data.images[Math.floor(Math.random() * data.images.length)];
          inspectArchiveImage(randImg);
        }
      });
    } else {
      video.src = '/api/sample_video';
    }
  });

  document.getElementById('fileUploadInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      video.src = url;
      video.play();
      setTimeout(captureAndInspectFrame, 500);
    }
  });
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

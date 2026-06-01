/**
 * DIGITAL BRAIN — Interactive Dashboard
 * =========================================
 * 3D visualization of the brain with Canvas 2D,
 * real-time WebSocket connection, and interactive panels.
 */

// ================================================================
// CONFIG
// ================================================================

const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`;
const API_URL = '/api';

const REGION_COLORS = {
  thalamus:         { h: 200, s: 80, l: 60, label: 'Thalamus' },
  visualCortex:     { h: 280, s: 70, l: 65, label: 'Visual Ctx' },
  auditoryCortex:   { h: 170, s: 70, l: 55, label: 'Auditory Ctx' },
  hippocampus:      { h: 45,  s: 85, l: 55, label: 'Hippocampus' },
  amygdala:         { h: 0,   s: 80, l: 60, label: 'Amygdala' },
  prefrontalCortex: { h: 220, s: 75, l: 65, label: 'Prefrontal Ctx' },
  wernicke:         { h: 130, s: 65, l: 55, label: 'Wernicke' },
  broca:            { h: 95,  s: 70, l: 50, label: 'Broca' },
};

// 3D positions of brain regions (x, y, z) normalized -1..1
const REGION_POSITIONS = {
  thalamus:         { x: 0,    y: 0,    z: 0,    size: 22 },
  visualCortex:     { x: 0,    y: 0.3,  z: -0.6, size: 30 },
  auditoryCortex:   { x: 0.55, y: -0.1, z: -0.1, size: 24 },
  hippocampus:      { x: 0.3,  y: -0.3, z: 0,    size: 24 },
  amygdala:         { x: 0.35, y: -0.15, z: 0.2,  size: 18 },
  prefrontalCortex: { x: 0,    y: -0.2, z: 0.65,  size: 35 },
  wernicke:         { x: -0.5, y: -0.1, z: 0.05,  size: 24 },
  broca:            { x: -0.4, y: -0.3, z: 0.35,  size: 24 },
};

// Connections between regions (for drawing axon lines)
const CONNECTIONS = [
  ['thalamus', 'visualCortex'],
  ['thalamus', 'auditoryCortex'],
  ['thalamus', 'wernicke'],
  ['thalamus', 'amygdala'],
  ['visualCortex', 'hippocampus'],
  ['visualCortex', 'amygdala'],
  ['auditoryCortex', 'hippocampus'],
  ['auditoryCortex', 'amygdala'],
  ['auditoryCortex', 'wernicke'],
  ['hippocampus', 'prefrontalCortex'],
  ['amygdala', 'prefrontalCortex'],
  ['prefrontalCortex', 'broca'],
  ['prefrontalCortex', 'thalamus'],
  ['prefrontalCortex', 'visualCortex'],
  ['amygdala', 'hippocampus'],
  ['wernicke', 'broca'],
  ['broca', 'hippocampus'],
];

// ================================================================
// STATE
// ================================================================

let ws = null;
let brainState = null;
let rotation = { x: -0.3, y: 0.7 };
let autoRotate = true;
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let particles = [];
let logEntries = [];

// ================================================================
// WEBSOCKET CONNECTION
// ================================================================

function connectWebSocket() {
  const dot = document.getElementById('connectionDot');
  const status = document.getElementById('brainStatus');

  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    status.textContent = 'Connection error';
    addLog('error', 'Could not connect to WebSocket');
    return;
  }

  ws.onopen = () => {
    dot.classList.add('connected');
    status.textContent = 'Connected — live';
    addLog('info', 'WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'state' || msg.type === 'init') {
        brainState = msg.data;
        updateDashboard(brainState);
      } else if (msg.type === 'thought') {
        addThought(msg.data);
      }
    } catch (e) {
      // Silently ignore parse errors
    }
  };

  ws.onclose = () => {
    dot.classList.remove('connected');
    status.textContent = 'Disconnected — reconnecting...';
    addLog('error', 'WebSocket disconnected');
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => {
    status.textContent = 'No connection to server';
  };
}

// ================================================================
// DASHBOARD UPDATE
// ================================================================

function updateDashboard(state) {
  if (!state) return;

  // Update tick counter and time
  const tickEl = document.querySelector('#tickCount .stat-value');
  const timeEl = document.querySelector('#timeDisplay .stat-value');
  if (tickEl) tickEl.textContent = (state.tickCount || 0).toLocaleString();
  if (timeEl) timeEl.textContent = formatTime(state.time || 0);

  // Update emotion chip
  const emotionChip = document.querySelector('#emotionChip .stat-value');
  if (state.emotion && emotionChip) {
    emotionChip.textContent = `${state.emotion.emoji} ${state.emotion.primaryEmotion}`;
  }

  // Update neuromodulator bars
  if (state.modulators) {
    for (const [type, level] of Object.entries(state.modulators)) {
      const bar = document.getElementById(`bar-${type}`);
      const val = document.getElementById(`val-${type}`);
      if (bar) bar.style.width = `${(level * 100).toFixed(0)}%`;
      if (val) val.textContent = level.toFixed(2);
    }
  }

  // Update emotion canvas
  if (state.emotion) {
    drawEmotionSpace(state.emotion);
    const label = document.getElementById('emotionLabel');
    if (label) label.textContent = `${state.emotion.emoji} ${state.emotion.primaryEmotion}`;
  }

  // Update region activity
  if (state.regions) {
    updateRegionActivity(state.regions);
  }

  // Update Broca chat
  updateBrocaChat(state);

  // Update learning panel (visual cortex)
  if (state.learning) {
    updateLearningPanel(state.learning);
  }

  // Update learning panel (hippocampus CA3)
  if (state.learningHippocampus) {
    updateLearningHippoPanel(state.learningHippocampus);
  }

  // Vocabulary acquisition panel
  updateVocabularyPanel(state.vocabulary, state.vocabCount);

  // Learning curve (growth of vocabulary + episodic memories over time)
  updateLearningCurve(state);

  // Consolidation section counters (were never wired)
  const memEl = document.getElementById('memoryCount');
  const vocEl = document.getElementById('vocabCount');
  if (memEl) memEl.textContent = state.memoriesCount || 0;
  if (vocEl) vocEl.textContent = (state.vocabulary && state.vocabulary.total) || state.vocabCount || 0;
}

// ================================================================
// VOCABULARY PANEL
// ================================================================

function updateVocabularyPanel(vocab, fallbackTotal) {
  const totalEl = document.getElementById('vocabTotal');
  const total = (vocab && vocab.total) || fallbackTotal || 0;
  if (totalEl) totalEl.textContent = total.toLocaleString();

  if (!vocab) return;

  const thrEl = document.getElementById('vocabThreshold');
  if (thrEl) thrEl.textContent = `(${vocab.threshold} reads to learn)`;

  // Words learned this session → chips (newest first, capped).
  const learned = vocab.learnedThisSession || [];
  const countEl = document.getElementById('vocabLearnedCount');
  if (countEl) countEl.textContent = learned.length;

  const chips = document.getElementById('vocabLearnedChips');
  if (chips) {
    if (learned.length === 0) {
      chips.innerHTML = '<span class="vocab-empty">none yet</span>';
    } else {
      chips.innerHTML = learned
        .slice(-24)
        .reverse()
        .map((w) => `<span class="vocab-chip">${escapeHtml(w)}</span>`)
        .join('');
    }
  }

  // Pending words → progress toward the learning threshold.
  const pend = document.getElementById('vocabPending');
  if (pend) {
    const pending = (vocab.pending || []).slice(0, 6);
    if (pending.length === 0) {
      pend.innerHTML = '<span class="vocab-empty">—</span>';
    } else {
      pend.innerHTML = pending
        .map((p) => {
          const pct = Math.min(100, (p.count / vocab.threshold) * 100);
          return (
            `<div class="vocab-pend-row">` +
            `<span class="vocab-pend-word">${escapeHtml(p.word)}</span>` +
            `<span class="vocab-pend-track"><span class="vocab-pend-fill" style="width:${pct.toFixed(0)}%"></span></span>` +
            `<span class="vocab-pend-count">${p.count}/${vocab.threshold}</span>` +
            `</div>`
          );
        })
        .join('');
    }
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// ================================================================
// LEARNING CURVE
// ================================================================

const curveHistory = { vocab: [], mem: [], cap: 120 };

function updateLearningCurve(state) {
  const canvas = document.getElementById('learningCurve');
  if (!canvas) return;

  const vocab = (state.vocabulary && state.vocabulary.total) || state.vocabCount || 0;
  const mem = state.memoriesCount || 0;

  curveHistory.vocab.push(vocab);
  curveHistory.mem.push(mem);
  if (curveHistory.vocab.length > curveHistory.cap) curveHistory.vocab.shift();
  if (curveHistory.mem.length > curveHistory.cap) curveHistory.mem.shift();

  const nowVocab = document.getElementById('curveVocabNow');
  const nowMem = document.getElementById('curveMemNow');
  if (nowVocab) nowVocab.textContent = vocab.toLocaleString();
  if (nowMem) nowMem.textContent = mem;

  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Each series auto-scaled to its OWN running max so both growth curves are
  // visible even though vocabulary (~hundreds) dwarfs memories (~tens).
  drawSeries(ctx, curveHistory.vocab, W, H, 'rgba(96, 165, 250, 0.95)');
  drawSeries(ctx, curveHistory.mem, W, H, 'rgba(74, 222, 128, 0.95)');
}

function drawSeries(ctx, data, W, H, color) {
  if (!data || data.length < 2) return;
  const pad = 3;
  const max = Math.max(1, ...data);
  const min = Math.min(...data);
  const span = Math.max(1, max - min);
  const n = data.length;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (i / (n - 1)) * (W - 2 * pad);
    const y = H - pad - ((data[i] - min) / span) * (H - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

// ================================================================
// LEARNING PANEL (Visual Cortex)
// ================================================================

let learnEngramCells = [];
function updateLearningPanel(m) {
  const stabilityBar = document.getElementById('learnStabilityBar');
  const stabilityVal = document.getElementById('learnStabilityVal');
  const weightBar = document.getElementById('learnWeightBar');
  const weightVal = document.getElementById('learnWeightVal');
  const activityBar = document.getElementById('learnActivityBar');
  const activityVal = document.getElementById('learnActivityVal');
  const engramSize = document.getElementById('learnEngramSize');
  const engramEl = document.getElementById('learnEngram');
  if (!stabilityBar) return;

  // Stability: 0-1 → 0-100%
  const stability = Math.max(0, Math.min(1, m.stability || 0));
  stabilityBar.style.width = `${(stability * 100).toFixed(0)}%`;
  stabilityVal.textContent = `${(stability * 100).toFixed(0)}%`;

  // Convergence: Σ|Δw| per tick. Inverse bar — the LOWER it is, the more consolidated.
  // Soft logarithmic scale against a reference (~5 = active learning).
  const dw = Math.max(0, m.weightChange || 0);
  const dwPct = Math.min(100, (dw / 5) * 100);
  weightBar.style.width = `${dwPct.toFixed(0)}%`;
  weightVal.textContent = dw.toFixed(2);

  // Cortical activity
  const act = Math.max(0, Math.min(1, (m.activity || 0) * 10)); // 10% ≈ full
  activityBar.style.width = `${(act * 100).toFixed(0)}%`;
  activityVal.textContent = `${((m.activity || 0) * 100).toFixed(1)}%`;

  // Engram: grid of cells, one per winning neuron.
  engramSize.textContent = m.engramSize || 0;
  const engram = m.engram || [];
  // Rebuild cells only if the count changes (avoids DOM thrashing)
  if (learnEngramCells.length !== engram.length) {
    engramEl.innerHTML = '';
    learnEngramCells = engram.map(() => {
      const cell = document.createElement('span');
      cell.className = 'engram-cell';
      engramEl.appendChild(cell);
      return cell;
    });
  }
  engram.forEach((id, i) => {
    const cell = learnEngramCells[i];
    if (cell) {
      cell.title = `neurona ${id}`;
      // Hue by ID so each neuron has a stable, recognizable color
      const hue = (id * 47) % 360;
      cell.style.background = `hsl(${hue}, 70%, 55%)`;
    }
  });
}

// ================================================================
// LEARNING PANEL (Hippocampus CA3)
// ================================================================

let hippoEngramCells = [];
function updateLearningHippoPanel(m) {
  const stabilityBar = document.getElementById('hippoStabilityBar');
  const stabilityVal = document.getElementById('hippoStabilityVal');
  const weightBar = document.getElementById('hippoWeightBar');
  const weightVal = document.getElementById('hippoWeightVal');
  const activityBar = document.getElementById('hippoActivityBar');
  const activityVal = document.getElementById('hippoActivityVal');
  const engramSize = document.getElementById('hippoEngramSize');
  const engramEl = document.getElementById('hippoEngram');
  const memCount = document.getElementById('hippoMemCount');
  if (!stabilityBar) return;

  // Attractor stability: 0-1 → 0-100%
  const stability = Math.max(0, Math.min(1, m.stability || 0));
  stabilityBar.style.width = `${(stability * 100).toFixed(0)}%`;
  stabilityVal.textContent = `${(stability * 100).toFixed(0)}%`;

  // Plasticity: Σ|Δw| per tick when imprinting a new episode.
  const dw = Math.max(0, m.weightChange || 0);
  const dwPct = Math.min(100, (dw / 5) * 100);
  weightBar.style.width = `${dwPct.toFixed(0)}%`;
  weightVal.textContent = dw.toFixed(2);

  // CA3 activity (sparse code ~2%)
  const act = Math.max(0, Math.min(1, (m.activity || 0) * 20)); // 5% ≈ full
  activityBar.style.width = `${(act * 100).toFixed(0)}%`;
  activityVal.textContent = `${((m.activity || 0) * 100).toFixed(1)}%`;

  // Stored episodes
  if (memCount) memCount.textContent = m.memoryCount || 0;

  // Recovered engram: grid of cells, one per attractor neuron.
  engramSize.textContent = m.engramSize || 0;
  const engram = m.engram || [];
  if (hippoEngramCells.length !== engram.length) {
    engramEl.innerHTML = '';
    hippoEngramCells = engram.map(() => {
      const cell = document.createElement('span');
      cell.className = 'engram-cell';
      engramEl.appendChild(cell);
      return cell;
    });
  }
  engram.forEach((id, i) => {
    const cell = hippoEngramCells[i];
    if (cell) {
      cell.title = `neurona ${id}`;
      const hue = (id * 47) % 360;
      cell.style.background = `hsl(${hue}, 70%, 55%)`;
    }
  });
}

function formatTime(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// ================================================================
// 3D BRAIN VISUALIZATION
// ================================================================

const brainCanvas = document.getElementById('brainCanvas');
const brainCtx = brainCanvas.getContext('2d');

function resizeBrainCanvas() {
  const rect = brainCanvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  brainCanvas.width = rect.width * dpr;
  brainCanvas.height = (rect.height - 80) * dpr;
  brainCanvas.style.width = `${rect.width}px`;
  brainCanvas.style.height = `${rect.height - 80}px`;
  brainCtx.scale(dpr, dpr);
}

function project3D(x, y, z, cx, cy, scale) {
  // Simple perspective projection
  const cosX = Math.cos(rotation.x);
  const sinX = Math.sin(rotation.x);
  const cosY = Math.cos(rotation.y);
  const sinY = Math.sin(rotation.y);

  // Rotate around Y
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  // Rotate around X
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;

  // Perspective
  const perspective = 3;
  const pScale = perspective / (perspective + z2 + 1.5);

  return {
    sx: cx + x1 * scale * pScale,
    sy: cy + y1 * scale * pScale,
    depth: z2,
    scale: pScale,
  };
}

function drawBrain() {
  const w = brainCanvas.width / (window.devicePixelRatio || 1);
  const h = brainCanvas.height / (window.devicePixelRatio || 1);
  const cx = w / 2;
  const cy = h / 2;
  const scale = Math.min(w, h) * 0.35;

  brainCtx.clearRect(0, 0, w, h);

  // Background glow
  const glow = brainCtx.createRadialGradient(cx, cy, 0, cx, cy, scale * 1.5);
  glow.addColorStop(0, 'rgba(56, 189, 248, 0.04)');
  glow.addColorStop(1, 'transparent');
  brainCtx.fillStyle = glow;
  brainCtx.fillRect(0, 0, w, h);

  if (autoRotate) {
    rotation.y += 0.005;
  }

  // Collect projected regions for depth sorting
  const projected = [];
  for (const [id, pos] of Object.entries(REGION_POSITIONS)) {
    const p = project3D(pos.x, pos.y, pos.z, cx, cy, scale);
    const color = REGION_COLORS[id];
    let activity = 0;
    if (brainState && brainState.regions && brainState.regions[id]) {
      const r = brainState.regions[id];
      activity = Math.min(1, (r.firingRate || 0) * 4); // same emphasis as the bars
    }
    projected.push({ id, pos, p, color, size: pos.size, activity });
  }

  // Sort by depth (far to near)
  projected.sort((a, b) => b.p.depth - a.p.depth);

  // Draw connections first (behind regions)
  drawConnections(projected, cx, cy, scale);

  // Draw particles
  updateParticles(cx, cy, scale);

  // Draw regions
  for (const region of projected) {
    drawRegion(region);
  }

  // Build legend
  buildLegend();

  requestAnimationFrame(drawBrain);
}

function drawConnections(projected, cx, cy, scale) {
  const regionMap = {};
  for (const r of projected) {
    regionMap[r.id] = r;
  }

  for (const [from, to] of CONNECTIONS) {
    const rFrom = regionMap[from];
    const rTo = regionMap[to];
    if (!rFrom || !rTo) continue;

    // Connection opacity based on activity
    let activity = Math.max(rFrom.activity, rTo.activity);
    let alpha = 0.08 + activity * 0.4;

    brainCtx.beginPath();
    brainCtx.moveTo(rFrom.p.sx, rFrom.p.sy);

    // Bezier curve for organic feel
    const midX = (rFrom.p.sx + rTo.p.sx) / 2;
    const midY = (rFrom.p.sy + rTo.p.sy) / 2 - 15;
    brainCtx.quadraticCurveTo(midX, midY, rTo.p.sx, rTo.p.sy);

    const gradient = brainCtx.createLinearGradient(
      rFrom.p.sx, rFrom.p.sy, rTo.p.sx, rTo.p.sy
    );
    const cFrom = rFrom.color;
    const cTo = rTo.color;
    gradient.addColorStop(0, `hsla(${cFrom.h}, ${cFrom.s}%, ${cFrom.l}%, ${alpha})`);
    gradient.addColorStop(1, `hsla(${cTo.h}, ${cTo.s}%, ${cTo.l}%, ${alpha})`);

    brainCtx.strokeStyle = gradient;
    brainCtx.lineWidth = 1 + activity * 3;
    brainCtx.stroke();

    // Spawn particles on active connections
    if (activity > 0.02 && Math.random() < activity * 0.3) {
      spawnParticle(rFrom, rTo);
    }
  }
}

function drawRegion(region) {
  const { p, color, size, activity, id } = region;
  const r = size * p.scale * (1 + activity * 0.5);

  // Outer glow
  if (activity > 0.01) {
    const glowR = r * (1.5 + activity * 2);
    const glow = brainCtx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, glowR);
    glow.addColorStop(0, `hsla(${color.h}, ${color.s}%, ${color.l}%, ${0.15 + activity * 0.3})`);
    glow.addColorStop(1, 'transparent');
    brainCtx.fillStyle = glow;
    brainCtx.beginPath();
    brainCtx.arc(p.sx, p.sy, glowR, 0, Math.PI * 2);
    brainCtx.fill();
  }

  // Main sphere (gradient for 3D look)
  const grad = brainCtx.createRadialGradient(
    p.sx - r * 0.25, p.sy - r * 0.25, r * 0.1,
    p.sx, p.sy, r
  );
  const baseLightness = color.l + activity * 15;
  grad.addColorStop(0, `hsla(${color.h}, ${color.s}%, ${Math.min(85, baseLightness + 15)}%, ${0.8 + activity * 0.2})`);
  grad.addColorStop(0.7, `hsla(${color.h}, ${color.s}%, ${baseLightness}%, ${0.6 + activity * 0.3})`);
  grad.addColorStop(1, `hsla(${color.h}, ${color.s - 10}%, ${baseLightness - 20}%, 0.3)`);

  brainCtx.fillStyle = grad;
  brainCtx.beginPath();
  brainCtx.arc(p.sx, p.sy, r, 0, Math.PI * 2);
  brainCtx.fill();

  // Border
  brainCtx.strokeStyle = `hsla(${color.h}, ${color.s}%, ${color.l + 10}%, ${0.3 + activity * 0.5})`;
  brainCtx.lineWidth = 1;
  brainCtx.stroke();

  // Label
  brainCtx.fillStyle = `rgba(241, 245, 249, ${0.5 + activity * 0.5})`;
  brainCtx.font = `${Math.max(9, 10 * p.scale)}px Inter, sans-serif`;
  brainCtx.textAlign = 'center';
  brainCtx.fillText(REGION_COLORS[id]?.label || id, p.sx, p.sy + r + 14);

  // Activity percentage
  if (activity > 0.001) {
    brainCtx.fillStyle = `hsla(${color.h}, ${color.s}%, 80%, 0.8)`;
    brainCtx.font = `bold ${Math.max(8, 9 * p.scale)}px JetBrains Mono, monospace`;
    brainCtx.fillText(`${(activity * 100).toFixed(0)}%`, p.sx, p.sy + 4);
  }
}

// ================================================================
// PARTICLES (spikes traveling along axons)
// ================================================================

function spawnParticle(from, to) {
  if (particles.length > 100) return; // Cap
  particles.push({
    fromX: from.p.sx,
    fromY: from.p.sy,
    toX: to.p.sx,
    toY: to.p.sy,
    t: 0,
    speed: 0.02 + Math.random() * 0.03,
    color: from.color,
    size: 2 + Math.random() * 2,
  });
}

function updateParticles(cx, cy, scale) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.t += p.speed;

    if (p.t >= 1) {
      particles.splice(i, 1);
      continue;
    }

    // Interpolate position with easing
    const t = p.t;
    const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const x = p.fromX + (p.toX - p.fromX) * ease;
    const y = p.fromY + (p.toY - p.fromY) * ease - Math.sin(t * Math.PI) * 10;

    const alpha = Math.sin(t * Math.PI); // Fade in/out

    // Draw particle
    brainCtx.fillStyle = `hsla(${p.color.h}, ${p.color.s}%, ${p.color.l + 20}%, ${alpha * 0.8})`;
    brainCtx.beginPath();
    brainCtx.arc(x, y, p.size, 0, Math.PI * 2);
    brainCtx.fill();

    // Glow trail
    const trailGlow = brainCtx.createRadialGradient(x, y, 0, x, y, p.size * 4);
    trailGlow.addColorStop(0, `hsla(${p.color.h}, ${p.color.s}%, ${p.color.l}%, ${alpha * 0.3})`);
    trailGlow.addColorStop(1, 'transparent');
    brainCtx.fillStyle = trailGlow;
    brainCtx.beginPath();
    brainCtx.arc(x, y, p.size * 4, 0, Math.PI * 2);
    brainCtx.fill();
  }
}

// ================================================================
// EMOTION CIRCUMPLEX VISUALIZATION
// ================================================================

function drawEmotionSpace(emotion) {
  const canvas = document.getElementById('emotionCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) * 0.4;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
  ctx.lineWidth = 1;

  // Concentric circles
  for (let i = 1; i <= 3; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * (i / 3), 0, Math.PI * 2);
    ctx.stroke();
  }

  // Axis lines
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Pleasure ↑', cx + r - 20, cy - 5);
  ctx.fillText('↓ Displeasure', cx - r + 25, cy - 5);
  ctx.fillText('High arousal', cx, cy - r + 12);
  ctx.fillText('Low arousal', cx, cy + r - 4);

  // Current emotion point
  if (emotion) {
    // Map valence (-1..1) to x, arousal (0..1) to y (inverted)
    const ex = cx + emotion.valence * r;
    const ey = cy - emotion.arousal * r * 0.8 + r * 0.1;

    // Glow
    const glow = ctx.createRadialGradient(ex, ey, 0, ex, ey, 20);
    glow.addColorStop(0, emotion.color + 'aa');
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(ex, ey, 20, 0, Math.PI * 2);
    ctx.fill();

    // Point
    ctx.fillStyle = emotion.color || '#38bdf8';
    ctx.beginPath();
    ctx.arc(ex, ey, 6, 0, Math.PI * 2);
    ctx.fill();

    // Border
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Emoji
    ctx.font = '16px serif';
    ctx.textAlign = 'center';
    ctx.fillText(emotion.emoji, ex, ey - 14);
  }
}

// ================================================================
// REGION ACTIVITY BARS
// ================================================================

function updateRegionActivity(regions) {
  const container = document.getElementById('regionActivity');
  if (!container) return;

  // Build rows if not exist. Each region shows TWO honest activity bars:
  //   • drive   — how much input signal the region is receiving (EMA, 0..1)
  //   • novelty — how much its firing PATTERN is changing (0..1)
  // We deliberately do NOT use `firingRate` here: with k-WTA sparse coding it is
  // ~constant by design (Wernicke/Broca/Prefrontal all pin at ~14%), so it can't
  // tell whether a region is engaged. drive + novelty actually react to interaction.
  if (container.children.length === 0) {
    for (const [id, color] of Object.entries(REGION_COLORS)) {
      const hsl = `hsl(${color.h}, ${color.s}%, ${color.l}%)`;
      const row = document.createElement('div');
      row.className = 'region-row';
      row.innerHTML = `
        <span class="region-name">${color.label}</span>
        <div class="region-bars">
          <div class="region-bar-track" title="drive — input signal received">
            <div class="region-bar-fill region-bar-drive" id="rdrive-${id}" style="background: ${hsl}"></div>
          </div>
          <div class="region-bar-track" title="novelty — firing-pattern change">
            <div class="region-bar-fill region-bar-nov" id="rnov-${id}" style="background: ${hsl}"></div>
          </div>
        </div>
        <div class="region-vals">
          <span class="region-rate" id="rdriveval-${id}">0%</span>
          <span class="region-rate region-rate-nov" id="rnovval-${id}">0%</span>
        </div>
      `;
      container.appendChild(row);
    }
  }

  // Update values. drive is scaled ×2 for the bar (typical range 0..0.3) so it is
  // visible, but the label shows the raw %. novelty already spans 0..1.
  for (const [id, data] of Object.entries(regions)) {
    const drive = Math.max(0, Math.min(1, data.drive || 0));
    const nov = Math.max(0, Math.min(1, data.novelty || 0));

    const driveBar = document.getElementById(`rdrive-${id}`);
    const driveVal = document.getElementById(`rdriveval-${id}`);
    if (driveBar) driveBar.style.width = `${Math.min(100, drive * 2 * 100).toFixed(0)}%`;
    if (driveVal) driveVal.textContent = `${(drive * 100).toFixed(0)}%`;

    const novBar = document.getElementById(`rnov-${id}`);
    const novVal = document.getElementById(`rnovval-${id}`);
    if (novBar) novBar.style.width = `${(nov * 100).toFixed(0)}%`;
    if (novVal) novVal.textContent = `${(nov * 100).toFixed(0)}%`;
  }
}

// ================================================================
// BRAIN LEGEND
// ================================================================

let legendBuilt = false;
function buildLegend() {
  if (legendBuilt) return;
  const container = document.getElementById('brainLegend');
  if (!container) return;

  for (const [id, color] of Object.entries(REGION_COLORS)) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background: hsl(${color.h}, ${color.s}%, ${color.l}%)"></div>${color.label}`;
    container.appendChild(item);
  }
  legendBuilt = true;
}

// ================================================================
// DRAWING CANVAS
// ================================================================

const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d');
let isDrawing = false;

drawCtx.fillStyle = '#111827';
drawCtx.fillRect(0, 0, 64, 64);

drawCanvas.addEventListener('pointerdown', (e) => {
  isDrawing = true;
  drawPixel(e);
});

drawCanvas.addEventListener('pointermove', (e) => {
  if (isDrawing) drawPixel(e);
});

drawCanvas.addEventListener('pointerup', () => { isDrawing = false; });
drawCanvas.addEventListener('pointerleave', () => { isDrawing = false; });

function drawPixel(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = 64 / rect.width;
  const scaleY = 64 / rect.height;
  const x = Math.floor((e.clientX - rect.left) * scaleX);
  const y = Math.floor((e.clientY - rect.top) * scaleY);

  drawCtx.fillStyle = '#f1f5f9';
  drawCtx.fillRect(x - 1, y - 1, 3, 3);
}

// ================================================================
// EVENT HANDLERS
// ================================================================

// Send text
document.getElementById('sendText')?.addEventListener('click', sendText);
document.getElementById('textInput')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendText();
});

function sendText() {
  const input = document.getElementById('textInput');
  const text = input.value.trim();
  if (!text) return;

  addLog('input', `Text: "${text}"`);
  input.value = '';

  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'input:text', data: { text } }));
  } else {
    fetch(`${API_URL}/input/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    .then(r => r.json())
    .then(result => {
      if (result.emotion) {
        addLog('emotion', `${result.emotion.emoji} ${result.emotion.primaryEmotion}`);
      }
      updateResponseBox(result);
    })
    .catch(err => addLog('error', `Error: ${err.message}`));
  }
}

// Send drawing
document.getElementById('sendDrawing')?.addEventListener('click', () => {
  const imageData = drawCtx.getImageData(0, 0, 64, 64);
  const pixels = [];

  // Convert to grayscale
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
    pixels.push(Math.round(gray));
  }

  addLog('input', 'Image sent (64×64)');

  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'input:image', data: { pixels, width: 64, height: 64 } }));
  } else {
    fetch(`${API_URL}/input/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels, width: 64, height: 64 }),
    })
    .then(r => r.json())
    .then(result => updateResponseBox(result))
    .catch(err => addLog('error', `Error: ${err.message}`));
  }
});

// Clear drawing
document.getElementById('clearDrawing')?.addEventListener('click', () => {
  drawCtx.fillStyle = '#111827';
  drawCtx.fillRect(0, 0, 64, 64);
});

// Inject neuromodulators
document.querySelectorAll('.inject-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const type = btn.dataset.type;
    addLog('info', `Injecting ${type} +0.2`);

    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'modulator', data: { type, amount: 0.2 } }));
    } else {
      fetch(`${API_URL}/modulator`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, amount: 0.2 }),
      }).catch(err => addLog('error', err.message));
    }
  });
});

// Brain canvas rotation
brainCanvas.addEventListener('pointerdown', (e) => {
  isDragging = true;
  dragStart = { x: e.clientX, y: e.clientY };
  autoRotate = false;
});

window.addEventListener('pointermove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragStart.x;
  const dy = e.clientY - dragStart.y;
  rotation.y += dx * 0.005;
  rotation.x += dy * 0.005;
  dragStart = { x: e.clientX, y: e.clientY };
});

window.addEventListener('pointerup', () => { isDragging = false; });

document.getElementById('toggleRotation')?.addEventListener('click', () => {
  autoRotate = !autoRotate;
});

document.getElementById('resetView')?.addEventListener('click', () => {
  rotation = { x: -0.3, y: 0.7 };
  autoRotate = true;
});

// ================================================================
// RESPONSE BOX
// ================================================================

function updateResponseBox(result) {
  const box = document.getElementById('responseBox');
  if (!box) return;

  let html = '';
  if (result.emotion) {
    html += `<div style="font-size: 1.2rem; margin-bottom: 0.5rem">${result.emotion.emoji} ${result.emotion.primaryEmotion}</div>`;
    html += `<div style="font-size: 0.75rem; color: var(--text-muted)">Valence: ${result.emotion.valence?.toFixed(2)} | Arousal: ${result.emotion.arousal?.toFixed(2)}</div>`;
  }
  if (result.broca && result.broca.words) {
    html += `<div style="font-size: 0.8rem; margin-top: 0.3rem">🗣️ ${result.broca.words.join(' ')}</div>`;
  }
  if (result.activeRegions && result.activeRegions.length > 0) {
    html += `<div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem">Active regions: ${result.activeRegions.join(', ')}</div>`;
  }
  if (result.processingTime) {
    html += `<div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.2rem">Processed in ${result.processingTime.toFixed(0)}ms</div>`;
  }

  box.innerHTML = html || '<p class="placeholder">No response</p>';
}

// ================================================================
// LOG
// ================================================================

function addLog(type, message) {
  const box = document.getElementById('logBox');
  if (!box) return;

  const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">${time}</span> <span class="log-msg">${message}</span>`;

  box.insertBefore(entry, box.firstChild);

  // Limit log entries
  while (box.children.length > 50) {
    box.removeChild(box.lastChild);
  }
}

// ================================================================
// STREAM OF CONSCIOUSNESS
// ================================================================

let lastThoughtText = null;

function addThought(thought) {
  const box = document.getElementById('thoughtBox');
  if (!box || !thought) return;

  // Skip empty / repeated consecutive thoughts to keep the stream meaningful.
  const words = Array.isArray(thought.words) ? thought.words : [];
  if (words.length === 0) return;
  if (thought.text === lastThoughtText) return;
  lastThoughtText = thought.text;

  // Drop the placeholder on first real thought.
  const empty = box.querySelector('.thought-empty');
  if (empty) empty.remove();

  const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const color = thought.color || '#94a3b8';

  const entry = document.createElement('div');
  entry.className = 'thought-entry';
  entry.innerHTML =
    `<span class="thought-time">${time}</span>` +
    `<span class="thought-emoji">${thought.emoji || '🧠'}</span>` +
    `<span class="thought-emotion" style="color:${color}">${(thought.emotion || '').toLowerCase()}</span>` +
    `<span class="thought-words">${words.join(' · ')}</span>`;

  box.insertBefore(entry, box.firstChild);

  // Fade older entries and cap the list.
  const entries = box.querySelectorAll('.thought-entry');
  entries.forEach((el, i) => { el.style.opacity = Math.max(0.25, 1 - i * 0.12); });
  while (box.children.length > 30) box.removeChild(box.lastChild);
}

// ================================================================
// WEBCAM
// ================================================================

let webcamStream = null;
let webcamInterval = null;

document.getElementById('toggleWebcam')?.addEventListener('click', async () => {
  const btn = document.getElementById('toggleWebcam');
  const status = document.getElementById('webcamStatus');
  const video = document.getElementById('webcamVideo');

  if (webcamStream) {
    // Stop
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    clearInterval(webcamInterval);
    webcamInterval = null;
    video.srcObject = null;
    btn.textContent = '▶ Enable';
    status.textContent = 'Inactive';
    addLog('info', '📷 Webcam disabled');
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 64, height: 64, facingMode: 'user' } });
    video.srcObject = webcamStream;
    btn.textContent = '⏹ Disable';
    status.textContent = 'Active — sending frames';
    addLog('info', '📷 Webcam enabled');

    const wCanvas = document.getElementById('webcamCanvas');
    const wCtx = wCanvas.getContext('2d');

    webcamInterval = setInterval(() => {
      wCtx.drawImage(video, 0, 0, 64, 64);
      const imageData = wCtx.getImageData(0, 0, 64, 64);
      const pixels = [];
      for (let i = 0; i < imageData.data.length; i += 4) {
        pixels.push(Math.round((imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3));
      }
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'input:image', data: { pixels, width: 64, height: 64 } }));
      }
    }, 500);
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    addLog('error', '📷 Webcam error: ' + err.message);
  }
});

// ================================================================
// MICROPHONE
// ================================================================

let micStream = null;
let micAnalyser = null;
let micInterval = null;

document.getElementById('toggleMic')?.addEventListener('click', async () => {
  const btn = document.getElementById('toggleMic');
  const status = document.getElementById('micStatus');

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
    micAnalyser = null;
    clearInterval(micInterval);
    micInterval = null;
    btn.textContent = '▶ Enable';
    status.textContent = 'Inactive';
    addLog('info', '🎤 Microphone disabled');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(micStream);
    micAnalyser = audioCtx.createAnalyser();
    micAnalyser.fftSize = 256;
    source.connect(micAnalyser);

    btn.textContent = '⏹ Disable';
    status.textContent = 'Active — analyzing audio';
    addLog('info', '🎤 Microphone enabled');

    const micCanvas = document.getElementById('micCanvas');
    const micCtx = micCanvas.getContext('2d');
    const bufferLength = micAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function drawMic() {
      if (!micAnalyser) return;
      requestAnimationFrame(drawMic);
      micAnalyser.getByteFrequencyData(dataArray);

      micCtx.fillStyle = '#0f172a';
      micCtx.fillRect(0, 0, micCanvas.width, micCanvas.height);

      const barWidth = micCanvas.width / bufferLength;
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * micCanvas.height;
        const hue = 170 + (i / bufferLength) * 60;
        micCtx.fillStyle = `hsla(${hue}, 70%, 55%, 0.8)`;
        micCtx.fillRect(i * barWidth, micCanvas.height - barHeight, barWidth - 1, barHeight);
      }
    }
    drawMic();

    // Send spectrograms every 250ms
    micInterval = setInterval(() => {
      if (!micAnalyser) return;
      micAnalyser.getByteFrequencyData(dataArray);
      const spectrogram = Array.from(dataArray).map(v => v / 255);

      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'input:audio', data: { spectrogram } }));
      }
    }, 250);
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    addLog('error', '🎤 Mic error: ' + err.message);
  }
});

// ================================================================
// BROCA CHAT
// ================================================================

function updateBrocaChat(state) {
  if (!state || !state.broca) return;
  const chat = document.getElementById('brocaChat');
  if (!chat) return;

  const text = state.broca.lastResponse;
  if (!text || text === '...') return;

  // Don't repeat the same message
  if (chat.dataset.lastMsg === text) return;
  chat.dataset.lastMsg = text;

  // Remove placeholder
  const placeholder = chat.querySelector('.placeholder');
  if (placeholder) placeholder.remove();

  const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const msgEl = document.createElement('div');
  msgEl.className = 'broca-msg';
  msgEl.innerHTML = `<div>${text}</div><div class="broca-time">${time}</div>`;
  chat.insertBefore(msgEl, chat.firstChild);

  while (chat.children.length > 20) chat.removeChild(chat.lastChild);
}

// ================================================================
// SLEEP BUTTON
// ================================================================

document.getElementById('sleepBtn')?.addEventListener('click', () => {
  const status = document.getElementById('sleepStatus');
  status.textContent = 'Sleeping...';
  addLog('info', '💤 Starting consolidation...');

  fetch(`${API_URL}/sleep`, { method: 'POST' })
    .then(r => r.json())
    .then(result => {
      status.textContent = 'Awake';
      addLog('info', `💤 Consolidation complete: ${result.memoriesReplayed || 0} memories`);
    })
    .catch(err => {
      status.textContent = 'Error';
      addLog('error', '💤 Error: ' + err.message);
    });
});

// ================================================================
// INITIALIZATION
// ================================================================

window.addEventListener('resize', resizeBrainCanvas);

// Start
resizeBrainCanvas();
drawBrain();
connectWebSocket();

addLog('info', '🧠 Dashboard started');

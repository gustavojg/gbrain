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

// Streaming cadence matches the server's per-client budget (see CLIENT_LIMITS
// in server-guards.ts): faster frames would just be dropped there.
const WEBCAM_FRAME_INTERVAL_MS = 2000;
// While someone speaks, spectrum frames stream every 200 ms so the ear's
// window holds the ORDER of the sounds (a syllable, a word); in silence, none.
const MIC_FRAME_INTERVAL_MS = 200;
// Mean normalized magnitude below which a mic frame is considered silence.
const MIC_SILENCE_THRESHOLD = 0.04;
// The TONE of a voice (envelope + pitch track) is sampled every 50 ms and sent
// once per utterance: it feeds the innate layer (warm / harsh), not the cortex.
const VOICE_FRAME_MS = 50;
const VOICE_ACTIVE_RMS = 0.015;
const VOICE_END_SILENCE_FRAMES = 8; // 400 ms of silence ends an utterance
const VOICE_MAX_FRAMES = 160;

const REGION_COLORS = {
  thalamus:         { h: 200, s: 80, l: 60, label: 'Thalamus' },
  visualCortex:     { h: 280, s: 70, l: 65, label: 'Visual Ctx' },
  auditoryCortex:   { h: 170, s: 70, l: 55, label: 'Auditory Ctx' },
  hippocampus:      { h: 45,  s: 85, l: 55, label: 'Hippocampus' },
  amygdala:         { h: 0,   s: 80, l: 60, label: 'Amygdala' },
  prefrontalCortex: { h: 220, s: 75, l: 65, label: 'Prefrontal Ctx' },
  wernicke:         { h: 130, s: 65, l: 55, label: 'Wernicke' },
  broca:            { h: 95,  s: 70, l: 50, label: 'Broca' },
  motorCortex:      { h: 25,  s: 85, l: 58, label: 'Vocal Motor' },
  handMotorCortex:  { h: 320, s: 70, l: 62, label: 'Hand Motor' },
  colorCortex:      { h: 50,  s: 90, l: 60, label: 'Colour Ctx' },
  partsCortex:      { h: 275, s: 70, l: 62, label: 'Parts Ctx (IT)' },
  nativeCortex:     { h: 10,  s: 80, l: 60, label: 'Native Ctx (C++)' },
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
  motorCortex:      { x: -0.15, y: 0.35, z: 0.3,  size: 20 },
  handMotorCortex:  { x: 0.2,   y: 0.4,  z: 0.25, size: 20 },
  colorCortex:      { x: 0.25,  y: 0.25, z: -0.55, size: 16 },
  partsCortex:      { x: -0.25, y: 0.15, z: -0.6,  size: 18 },
  nativeCortex:     { x: 0.0,   y: -0.35, z: -0.7, size: 20 },
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
  ['auditoryCortex', 'motorCortex'],
  ['thalamus', 'handMotorCortex'],
  ['thalamus', 'colorCortex'],
  ['thalamus', 'partsCortex'],
  ['thalamus', 'nativeCortex'],
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
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      return; // Not JSON — ignore
    }
    // Rendering errors must surface (console) instead of silently freezing the UI.
    if (msg.type === 'state' || msg.type === 'init') {
      brainState = msg.data;
      updateDashboard(brainState);
    } else if (msg.type === 'thought') {
      addThought(msg.data);
    } else if (msg.type === 'vocalization' && msg.data) {
      playVocalization(msg.data);
    } else if (msg.type === 'drawing' && msg.data) {
      showBrainDrawing(msg.data);
    } else if (msg.type === 'writing' && msg.data) {
      showBrainWriting(msg.data);
    } else if (msg.type === 'imagination' && msg.data) {
      showImagination(msg.data);
    } else if (msg.type === 'lesson' && msg.data) {
      showLessonProgress(msg.data);
    } else if (msg.type === 'practice' && msg.data) {
      showPracticeProgress(msg.data);
    } else if (msg.type === 'affect' && msg.data) {
      showAffect(msg.data);
    } else if (msg.type === 'notice' && msg.data) {
      addLog('error', `Server: ${msg.data.message}`);
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
  updatePerceptionPanel(state.recognition);
  updateRecallRow(state.association);
  updateInnateRow(state.innate);
  updateMotivationRow(state.motivation);
  updateExpectationRow(state.sequence);
  updateWorkingMemory(state.workingMemory);

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

// ================================================================
// PERCEPTION PANEL (learning by exposure)
// ================================================================

const lastPerceptLogged = { visual: null, colour: null, auditory: null };

function updatePerceptionPanel(recognition) {
  if (!recognition) return;
  renderPercept('perceptVisual', 'visual', recognition.visual, recognition.visualCategories, 'seen', recognition.object ? ` · object ${recognition.object.label}${recognition.object.isNew ? ' (new)' : ''} · ${Number(recognition.partsKnown)} parts` : '');
  renderPercept('perceptColour', 'colour', recognition.colour, recognition.colourCategories, 'seen');
  renderPercept('perceptAuditory', 'auditory', recognition.auditory, recognition.auditoryCategories, 'heard');
}

let lastRecallLogged = null;

/** What the last percept brought back from memory: "Visual-1 → “cruz” · 82%". */
const TONE_FACES = { warm: '😊 warm', neutral: '😐 neutral', harsh: '😠 harsh' };
let lastVoiceShown = 0;

function updateInnateRow(innate) {
  const what = document.querySelector('#perceptTone .percept-what');
  if (!what || !innate) return;
  const v = innate.lastVoice;
  if (!v) {
    what.textContent = innate.affectiveWords > 0 || innate.conditionedCues > 0
      ? `no voice heard yet · ${Number(innate.affectiveWords)} words and ${Number(innate.conditionedCues)} things carry a feeling`
      : 'no voice heard yet — talk to it: the tone is what it reads';
    return;
  }
  what.classList.remove('percept-empty');
  const pct = Math.round(Math.abs(Number(v.valence)) * 100);
  what.innerHTML =
    `<span class="percept-label">${TONE_FACES[v.kind] || v.kind}</span> ${pct}%` +
    (v.startle ? ' <span class="percept-badge is-new">startled</span>' : '') +
    (v.judged ? ` <span class="percept-badge ${v.valence > 0 ? 'is-known' : 'is-new'}">${v.valence > 0 ? 'approved' : 'corrected'} what it recalled</span>` : '') +
    `<span class="percept-total">${Number(innate.affectiveWords)} words · ${Number(innate.conditionedCues)} things carry a feeling · sleep pressure ${Math.round(Number(innate.sleepPressure) * 100)}%</span>`;
  if (v.serial !== lastVoiceShown) {
    lastVoiceShown = v.serial;
    addLog('info', `🗣️ Heard a ${v.kind} voice (${pct}%)${v.judged ? v.valence > 0 ? ' — took it as approval' : ' — took it as a correction' : ''}`);
  }
}

let lastDopamineShown = 0;

function updateMotivationRow(m) {
  const what = document.querySelector('#perceptDrives .percept-what');
  if (!what || !m) return;
  const pct = (x) => `${Math.round(Number(x) * 100)}%`;
  const e = m.lastEvent;
  let last = '';
  if (e) {
    const sign = e.error >= 0 ? '+' : '−';
    const what = { novelty: 'new', progress: 'learning', external: e.reward >= 0 ? 'praise' : 'reprimand', omission: 'expected, nothing came', imagination: 'imagined' }[e.kind] || e.kind;
    last = ` · last: ${what}${e.key ? ` (${escapeHtml(e.key.replace(/^(visual|auditory|word|activity):/, ''))})` : ''} dopamine ${sign}${Math.abs(e.error).toFixed(2)}`;
  }
  what.classList.remove('percept-empty');
  what.innerHTML =
    `curious ${pct(m.drives.curiosity)} · bored ${pct(m.drives.boredom)} · lonely ${pct(m.drives.contact)}` +
    `<span class="percept-total">babble ${Number(m.activityValues.babble).toFixed(2)} · scribble ${Number(m.activityValues.scribble).toFixed(2)} · daydream ${Number(m.activityValues.daydream ?? 0).toFixed(2)} (worth of each activity) · ${Number(brainState?.habits?.count ?? 0)} habits${escapeHtml(last)}</span>`;
}

function updateWorkingMemory(slots) {
  const box = document.getElementById('wmSlots');
  if (!box || !Array.isArray(slots)) return;
  const cells = box.querySelectorAll('.wm-slot');
  const sorted = slots.slice().sort((a, b) => Number(b.priority) - Number(a.priority));
  cells.forEach((cell, i) => {
    const slot = sorted[i];
    if (slot) {
      cell.textContent = String(slot.label).slice(0, 12);
      cell.title = `held with priority ${Math.round(Number(slot.priority) * 100)}%, ${Number(slot.age)} ticks`;
      cell.classList.add('active');
      cell.classList.remove('empty');
      cell.style.opacity = String(0.45 + 0.55 * Math.min(1, Number(slot.priority)));
    } else {
      cell.textContent = String(i + 1);
      cell.title = '';
      cell.classList.remove('active');
      cell.classList.add('empty');
      cell.style.opacity = '';
    }
  });
}

function updateExpectationRow(sequence) {
  const what = document.querySelector('#perceptNext .percept-what');
  if (!what || !sequence) return;
  const e = sequence.expectation;
  if (!e) {
    what.textContent = sequence.transitions > 0
      ? `nothing in particular · ${Number(sequence.transitions)} orders learned`
      : 'nothing yet — show it things in the same order a few times';
    return;
  }
  what.classList.remove('percept-empty');
  what.innerHTML = `<span class="percept-label">${escapeHtml(String(e.key).replace(/^(visual|auditory|lexical):/, ''))}</span> ` +
    `<span class="percept-badge is-known">${Math.round(Number(e.probability) * 100)}%</span> in ~${(Number(e.expectedInTicks) / 10).toFixed(0)} s` +
    `<span class="percept-total">${Number(sequence.transitions)} orders learned${sequence.lastTransition ? ` · last: ${escapeHtml(sequence.lastTransition.from.replace(/^[a-z]+:/, ''))} → ${escapeHtml(sequence.lastTransition.to.replace(/^[a-z]+:/, ''))}` : ''}</span>`;
}

function showAffect(d) {
  if (d.kind === 'dopamine' && d.event) {
    const e = d.event;
    if (Math.abs(Number(e.error)) < 0.15 || e.kind === 'progress' && e.key && e.key.startsWith('activity:')) return;
    const name = e.key ? escapeHtml(String(e.key).replace(/^(imagined|foreseen):/, '').replace(/(^|\+)(visual|colour|auditory|word|activity):/g, '$1')) : 'that';
    const line = e.kind === 'novelty' ? `✨ ${name} is new to it` :
      e.kind === 'progress' ? `📈 It is getting better at ${name}` :
      e.kind === 'imagination' ? (String(e.key || '').startsWith('foreseen:') ? `🔮 It had imagined this: ${name}` : `💭 Imagining something new: ${name}`) :
      e.kind === 'omission' ? `😕 It expected praise for ${name} and nothing came` :
      Number(e.error) > 0 ? `🎁 Better than it expected for ${name}` : `😞 Worse than it expected for ${name}`;
    addLog('emotion', `${line} (dopamine ${Number(e.error) >= 0 ? '+' : ''}${Number(e.error).toFixed(2)})`);
    return;
  }
  if (d.kind === 'question-learned') { addLog('info', `❓ “${escapeHtml(String(d.question).replace(/^lexical:/, ''))}” asks for the ${escapeHtml(String(d.modality))} of things (${Number(d.known)} questions known)`); return; }
  if (d.kind === 'answer') return; // the writing line says it
  if (d.kind === 'saccade') { addLog('info', `👀 Looking at thing ${Number(d.index)} of ${Number(d.count)}`); return; }
  if (d.kind === 'foreseen') { addLog('emotion', `🔮 It had imagined ${escapeHtml((d.sources || []).map((s) => String(s).replace(/^[a-z]+:/, '')).join(' + '))} — and here it is`); return; }
  if (d.kind === 'startle') addLog('emotion', '😳 Startled by a sudden loud sound');
  else if (d.kind === 'looming') addLog('emotion', '😨 Something is coming closer fast');
  else if (d.kind === 'face') addLog('emotion', `🙂 That looks like a face (${Math.round(Number(d.match) * 100)}%)`);
  else if (d.kind === 'conditioned') addLog('emotion', `${Number(d.valence) < 0 ? '😟' : '😌'} ${escapeHtml(String(d.label))} brings back a feeling (${Number(d.valence) < 0 ? 'unease' : 'comfort'})`);
}

function updateRecallRow(association) {
  const what = document.querySelector('#perceptRecall .percept-what');
  const r = association && association.lastRecall;
  if (!what || !r) return;

  const parts = [];
  if (r.words && r.words.length > 0) parts.push(`“${r.words.map((w) => escapeHtml(w.word)).join(' ')}”`);
  if (r.visual) parts.push(escapeHtml(r.visual.label));
  if (r.colour) parts.push(escapeHtml(r.colour.label));
  if (r.auditory) parts.push(escapeHtml(r.auditory.label));
  const recalled = parts.length > 0 ? parts.join(' + ') : 'something it cannot name yet';
  const pct = Math.round(Number(r.confidence) * 100);

  what.classList.remove('percept-empty');
  what.innerHTML =
    `<span class="percept-label">${escapeHtml(r.cue.label)}</span> → ${recalled}` +
    `<span class="percept-badge ${r.confident ? 'is-known' : 'is-new'}">${pct}%${r.confident ? '' : ' · unsure'}</span>` +
    `<span class="percept-total">${Number(association.bindings)} shared experience${association.bindings === 1 ? '' : 's'}</span>`;

  const key = `${r.timestamp}:${r.cue.label}`;
  if (lastRecallLogged !== key && r.confident && parts.length > 0) {
    lastRecallLogged = key;
    addLog('info', `🔗 ${r.cue.label} reminds it of ${parts.join(' + ').replace(/<[^>]*>/g, '')} (${pct}%)`);
  }
}

function renderPercept(rowId, sense, r, categories, verb, extra = '') {
  const what = document.querySelector(`#${rowId} .percept-what`);
  if (!what || !r) return;

  const surprise = typeof r.surprise === 'number' ? ` · surprise ${Math.round(Number(r.surprise) * 100)}%` : '';
  const badge = r.isNew
    ? `<span class="percept-badge is-new">new${surprise}</span>`
    : `<span class="percept-badge is-known">${verb} ×${Number(r.exposures)} · ${Math.round(Number(r.familiarity) * 100)}% match${surprise}</span>`;
  what.classList.remove('percept-empty');
  what.innerHTML =
    `<span class="percept-label">${escapeHtml(r.label)}</span>${badge}` +
    `<span class="percept-total">${Number(categories)} categor${categories === 1 ? 'y' : 'ies'}${escapeHtml(extra)}</span>`;

  // Log each recognition once (the state stream repeats the last one).
  const key = `${r.id}:${r.exposures}`;
  if (lastPerceptLogged[sense] !== key) {
    lastPerceptLogged[sense] = key;
    const icon = { visual: '👁️', colour: '🎨', auditory: '👂' }[sense] || '👁️';
    addLog('info', r.isNew ? `${icon} Something new → ${r.label}` : `${icon} Recognized ${r.label} (${verb} ×${r.exposures})`);
  }
}

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
  if (countEl) countEl.textContent = vocab.learnedCount ?? learned.length;

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
  const panel = brainCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const styles = getComputedStyle(panel);
  const width = panel.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
  const legend = document.getElementById('brainLegend');
  const header = panel.querySelector('.panel-header');
  const used = (header?.offsetHeight || 0) + (legend?.offsetHeight || 0)
    + parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom) + 24;
  const height = Math.max(160, panel.clientHeight - used);
  if (width <= 0) return;
  brainCanvas.width = Math.round(width * dpr);
  brainCanvas.height = Math.round(height * dpr);
  brainCanvas.style.width = `${width}px`;
  brainCanvas.style.height = `${height}px`;
  brainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
    // Map valence (-1..1) to x and arousal (0..1, amygdala) to the whole
    // vertical axis: 0 at the bottom (low arousal), 1 at the top.
    const valence = Math.max(-1, Math.min(1, Number(emotion.valence) || 0));
    const arousal = Math.max(0, Math.min(1, Number(emotion.arousal) || 0));
    const ex = cx + valence * r * 0.9;
    const ey = cy + (1 - 2 * arousal) * r * 0.9;

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
    // drivePeak: a wave crosses the early regions faster than the 2 Hz state stream.
    const drive = Math.max(0, Math.min(1, data.drivePeak ?? data.drive ?? 0));
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
let lastInk = null;
let inkColour = '#f1f5f9';
// The strokes of the drawing in progress: how it was made, for the hand to learn the gesture.
let drawStrokes = [];
let drawStroke = null;

document.querySelectorAll('#inkPalette .ink-swatch').forEach((swatch) => {
  swatch.addEventListener('click', () => {
    inkColour = swatch.dataset.ink;
    document.querySelectorAll('#inkPalette .ink-swatch').forEach((s) => s.classList.toggle('is-selected', s === swatch));
  });
});

/** The pixels of a 64×64 canvas as the brain takes them: grey levels and, apart, their colour. */
function pixelsOf(ctx) {
  const data = ctx.getImageData(0, 0, 64, 64).data;
  const pixels = [];
  const rgb = [];
  for (let i = 0; i < data.length; i += 4) {
    pixels.push(Math.round((data[i] + data[i + 1] + data[i + 2]) / 3));
    rgb.push(data[i], data[i + 1], data[i + 2]);
  }
  return { pixels, rgb };
}

drawCtx.fillStyle = '#111827';
drawCtx.fillRect(0, 0, 64, 64);

// Mouse, pen or finger: one code path. `touch-action: none` (styles.css) keeps the
// page from scrolling under the finger; pointer capture keeps the stroke alive
// when it leaves the canvas; a fast swipe is interpolated so the 64×64 grid gets a
// continuous line rather than dots.
drawCanvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 && e.pointerType === 'mouse') return;
  e.preventDefault();
  isDrawing = true;
  lastInk = null;
  drawStroke = { points: [], start: performance.now() };
  try { drawCanvas.setPointerCapture(e.pointerId); } catch { /* not supported */ }
  drawPixel(e);
});

drawCanvas.addEventListener('pointermove', (e) => {
  if (isDrawing) drawPixel(e);
});

const endStroke = () => {
  isDrawing = false;
  lastInk = null;
  if (drawStroke && drawStroke.points.length > 0) {
    drawStrokes.push({ points: drawStroke.points, durationMs: Math.round(performance.now() - drawStroke.start) });
    if (drawStrokes.length > 32) drawStrokes.shift();
  }
  drawStroke = null;
};
drawCanvas.addEventListener('pointerup', endStroke);
drawCanvas.addEventListener('pointercancel', endStroke);
drawCanvas.addEventListener('lostpointercapture', endStroke);

function drawPixel(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) * (64 / rect.width));
  const y = Math.floor((e.clientY - rect.top) * (64 / rect.height));
  drawCtx.fillStyle = inkColour;
  if (drawStroke && drawStroke.points.length < 512 && (!lastInk || lastInk.x !== x || lastInk.y !== y)) drawStroke.points.push([x, y]);
  if (lastInk) {
    const steps = Math.max(Math.abs(x - lastInk.x), Math.abs(y - lastInk.y));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      drawCtx.fillRect(Math.round(lastInk.x + (x - lastInk.x) * t) - 1, Math.round(lastInk.y + (y - lastInk.y) * t) - 1, 3, 3);
    }
  } else {
    drawCtx.fillRect(x - 1, y - 1, 3, 3);
  }
  lastInk = { x, y };
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
  const { pixels, rgb } = pixelsOf(drawCtx);
  addLog('input', 'Image sent (64×64, with colour)');

  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'input:image', data: { pixels, rgb, width: 64, height: 64 } }));
  } else {
    fetch(`${API_URL}/input/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels, rgb, width: 64, height: 64 }),
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
  drawStrokes = [];
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
  e.preventDefault();
  isDragging = true;
  dragStart = { x: e.clientX, y: e.clientY };
  autoRotate = false;
  try { brainCanvas.setPointerCapture(e.pointerId); } catch { /* not supported */ }
});

brainCanvas.addEventListener('pointermove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - dragStart.x;
  const dy = e.clientY - dragStart.y;
  rotation.y += dx * 0.005;
  rotation.x += dy * 0.005;
  dragStart = { x: e.clientX, y: e.clientY };
});

const endDrag = () => { isDragging = false; };
brainCanvas.addEventListener('pointerup', endDrag);
brainCanvas.addEventListener('pointercancel', endDrag);
brainCanvas.addEventListener('lostpointercapture', endDrag);

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
    html += `<div style="font-size: 1.2rem; margin-bottom: 0.5rem">${escapeHtml(result.emotion.emoji)} ${escapeHtml(result.emotion.primaryEmotion)}</div>`;
    html += `<div style="font-size: 0.75rem; color: var(--text-muted)">Valence: ${Number(result.emotion.valence).toFixed(2)} | Arousal: ${Number(result.emotion.arousal).toFixed(2)}</div>`;
  }
  if (result.broca && result.broca.words) {
    html += `<div style="font-size: 0.8rem; margin-top: 0.3rem">🗣️ ${escapeHtml(result.broca.words.join(' '))}</div>`;
  }
  if (result.activeRegions && result.activeRegions.length > 0) {
    html += `<div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem">Active regions: ${escapeHtml(result.activeRegions.join(', '))}</div>`;
  }
  if (result.processingTime) {
    html += `<div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.2rem">Processed in ${Number(result.processingTime).toFixed(0)}ms</div>`;
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
  entry.innerHTML = `<span class="log-time">${time}</span> <span class="log-msg">${escapeHtml(message)}</span>`;

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
  // The color lands in a style attribute: accept only a plain hex color.
  const color = /^#[0-9a-f]{3,8}$/i.test(thought.color || '') ? thought.color : '#94a3b8';

  const entry = document.createElement('div');
  entry.className = 'thought-entry';
  entry.innerHTML =
    `<span class="thought-time">${time}</span>` +
    `<span class="thought-emoji">${escapeHtml(thought.emoji || '🧠')}</span>` +
    `<span class="thought-emotion" style="color:${color}">${escapeHtml(String(thought.emotion || '').toLowerCase())}</span>` +
    `<span class="thought-words">${escapeHtml(words.join(' · '))}</span>`;

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
let webcamOpening = false;

document.getElementById('toggleWebcam')?.addEventListener('click', async () => {
  const btn = document.getElementById('toggleWebcam');
  const status = document.getElementById('webcamStatus');
  const video = document.getElementById('webcamVideo');

  if (webcamOpening) return;
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
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = 'Not available (needs https or localhost)';
    return;
  }

  webcamOpening = true;
  btn.disabled = true;
  status.textContent = 'Asking for the camera…';
  try {
    // A normal-resolution stream for the preview; the frame the brain gets is
    // reduced to its 64×64 retina on the canvas beside it.
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } });
    video.srcObject = webcamStream;
    btn.textContent = '⏹ Disable';
    status.textContent = 'Active — sending frames';
    addLog('info', '📷 Webcam enabled');

    const wCanvas = document.getElementById('webcamCanvas');
    const wCtx = wCanvas.getContext('2d');

    webcamInterval = setInterval(() => {
      // Centre crop to a square, then reduce to the retina.
      const side = Math.min(video.videoWidth, video.videoHeight) || 64;
      const sx = ((video.videoWidth || side) - side) / 2, sy = ((video.videoHeight || side) - side) / 2;
      wCtx.drawImage(video, sx, sy, side, side, 0, 0, 64, 64);
      const { pixels, rgb } = pixelsOf(wCtx);
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'input:image', data: { pixels, rgb, width: 64, height: 64 } }));
      }
    }, WEBCAM_FRAME_INTERVAL_MS);
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    addLog('error', '📷 Webcam error: ' + err.message);
  } finally {
    webcamOpening = false;
    btn.disabled = false;
  }
});

// ================================================================
// MICROPHONE
// ================================================================

let micStream = null;
let micAnalyser = null;
let micInterval = null;
let voiceInterval = null;
let micAudioCtx = null;
let micOpening = false;
let micSpeaking = false;

function rmsOf(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/** Fundamental frequency by normalized autocorrelation (70–500 Hz); 0 if unvoiced. */
/**
 * The pitch of a frame by normalized autocorrelation, 70–500 Hz. Each lag is
 * normalized by the energy of the two stretches actually compared (dividing
 * by the whole frame's energy, as before, capped a low male voice at
 * r ≈ 0.5 and read most of it as unvoiced). Prefers the longest lag among
 * near-equal peaks, so a strong second harmonic does not halve the period.
 */
function pitchOf(samples, sampleRate) {
  const n = samples.length;
  const minLag = Math.floor(sampleRate / 500);
  const maxLag = Math.min(n >> 1, Math.ceil(sampleRate / 70));
  let bestLag = 0;
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0, e1 = 0, e2 = 0;
    for (let i = 0; i + lag < n; i++) { const a = samples[i], b = samples[i + lag]; sum += a * b; e1 += a * a; e2 += b * b; }
    const r = e1 > 0 && e2 > 0 ? sum / Math.sqrt(e1 * e2) : 0;
    if (r > best) { best = r; bestLag = lag; }
  }
  if (best < 0.6 || bestLag === 0) return 0;
  // Octave check: if double the lag correlates almost as well, that is the true period.
  const twice = bestLag * 2;
  if (twice <= maxLag) {
    let sum = 0, e1 = 0, e2 = 0;
    for (let i = 0; i + twice < n; i++) { const a = samples[i], b = samples[i + twice]; sum += a * b; e1 += a * a; e2 += b * b; }
    const r2 = e1 > 0 && e2 > 0 ? sum / Math.sqrt(e1 * e2) : 0;
    if (r2 >= best * 0.9) bestLag = twice;
  }
  return sampleRate / bestLag;
}

document.getElementById('toggleMic')?.addEventListener('click', async () => {
  const btn = document.getElementById('toggleMic');
  const status = document.getElementById('micStatus');

  if (micOpening) return;
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
    micAnalyser = null;
    clearInterval(micInterval);
    micInterval = null;
    clearInterval(voiceInterval);
    voiceInterval = null;
    micAudioCtx?.close().catch(() => {});
    micAudioCtx = null;
    btn.textContent = '▶ Enable';
    status.textContent = 'Inactive';
    addLog('info', '🎤 Microphone disabled');
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = 'Not available (needs https or localhost)';
    return;
  }

  micOpening = true;
  btn.disabled = true;
  status.textContent = 'Asking for the microphone…';
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtor();
    micAudioCtx = audioCtx;
    await audioCtx.resume(); // iOS creates it suspended
    const source = audioCtx.createMediaStreamSource(micStream);
    micAnalyser = audioCtx.createAnalyser();
    micAnalyser.fftSize = 2048; // ~23 Hz per bin at 48 kHz; and 43 ms of waveform, four periods of a low male voice, for the pitch
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

    // The tone of the voice: envelope + pitch every 50 ms, sent per utterance.
    const timeData = new Float32Array(micAnalyser.fftSize);
    let contour = { rms: [], f0: [] };
    let silentFrames = 0;
    let ownVoiceAtStart = false;
    micSpeaking = false;
    voiceInterval = setInterval(() => {
      if (!micAnalyser) return;
      micAnalyser.getFloatTimeDomainData(timeData);
      const rms = rmsOf(timeData);
      const active = rms >= VOICE_ACTIVE_RMS;
      micSpeaking = active;
      if (contour.rms.length === 0) {
        if (!active) return;
        ownVoiceAtStart = performance.now() < ownVoiceUntil;
      }
      contour.rms.push(Math.round(rms * 1000) / 1000);
      contour.f0.push(active ? Math.round(pitchOf(timeData, audioCtx.sampleRate)) : 0);
      silentFrames = active ? 0 : silentFrames + 1;
      const done = silentFrames >= VOICE_END_SILENCE_FRAMES || contour.rms.length >= VOICE_MAX_FRAMES;
      if (!done) return;
      const utterance = contour;
      contour = { rms: [], f0: [] };
      silentFrames = 0;
      // Its own voice, played through the speakers, is not somebody talking to it.
      if (ownVoiceAtStart) return;
      if (utterance.rms.length - VOICE_END_SILENCE_FRAMES < 3) return; // a click, not a voice
      sendControl('input:voice', { rms: utterance.rms, f0: utterance.f0, frameMs: VOICE_FRAME_MS }, 'input/voice');
    }, VOICE_FRAME_MS);

    // Send spectrograms periodically — but never silence: each frame costs the
    // server a full perception, and silence carries no information.
    micInterval = setInterval(() => {
      if (!micAnalyser) return;
      micAnalyser.getByteFrequencyData(dataArray);
      const spectrogram = Array.from(dataArray).map(v => v / 255);
      const energy = spectrogram.reduce((sum, v) => sum + v, 0) / spectrogram.length;
      if (energy < MIC_SILENCE_THRESHOLD && !micSpeaking) return;
      // The brain already hears its own voice internally: what the mic picks up
      // from the speakers must not come back as somebody else's sound.
      if (performance.now() < ownVoiceUntil) return;

      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'input:audio', data: { spectrogram, sampleRate: audioCtx.sampleRate } }));
      }
    }, MIC_FRAME_INTERVAL_MS);
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    addLog('error', '🎤 Mic error: ' + err.message);
    micStream?.getTracks().forEach(t => t.stop());
    micStream = null;
    micAudioCtx?.close().catch(() => {});
    micAudioCtx = null;
  } finally {
    micOpening = false;
    btn.disabled = false;
  }
});

// ================================================================
// VOICE — the brain's vocal tract, rendered with WebAudio
// ================================================================
// Same articulatory model as src/core/voice/vocal-tract.ts: a voiced source
// shaped by two formants. The brain decides F1/F2; this only makes them
// audible. What the brain HEARS of itself is the model's spectrum, not this
// playback, so nothing here changes what it learns — only what you hear.
//
// The voice is a small child's: a fundamental of ~260 Hz (an infant's 380 Hz
// left too few harmonics under the formants to hear a vowel — it whistled),
// a natural slight rise-and-fall contour and a little vibrato, formants as
// wide as the model's (a narrow, fixed-pitch buzz is what sounds like a
// robot) but with steep skirts (two filter stages each), a weak third
// formant and a breath of aspiration noise.

let voiceCtx = null;
let voiceEnabled = false;
let ownVoiceUntil = 0;
/** A small child's fundamental frequency (adults: 100–250 Hz; infants: 300–500 Hz). */
const VOICE_PITCH_HZ = 260;
/** Formant bandwidths, as the vocal tract model's (FWHM of its Gaussian peaks: 2.355 × 90 and × 120 Hz). */
const FORMANT_BANDWIDTH_HZ = { f1: 212, f2: 283, f3: 400 };
/** The third formant: fixed, weak; it makes the vowel a voice rather than two whistles. */
const F3_HZ = 3000;

document.getElementById('toggleVoice')?.addEventListener('click', () => {
  const btn = document.getElementById('toggleVoice');
  const status = document.getElementById('voiceStatus');
  voiceEnabled = !voiceEnabled;

  if (voiceEnabled) {
    // Browsers only allow audio to start from a user gesture — this click.
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (!voiceCtx && AudioCtor) voiceCtx = new AudioCtor();
    voiceCtx?.resume();
  }
  btn.textContent = voiceEnabled ? '⏹ Disable' : '▶ Enable';
  btn.setAttribute('aria-pressed', String(voiceEnabled));
  status.textContent = voiceEnabled ? 'Babbling — it learns what its commands sound like' : 'Silent';
  addLog('info', voiceEnabled ? '🗣️ Voice enabled: babbling + imitation' : '🗣️ Voice disabled');

  const voice = { babble: voiceEnabled, imitate: voiceEnabled };
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'voice', data: voice }));
  } else {
    fetch(`${API_URL}/voice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(voice),
    }).catch(err => addLog('error', err.message));
  }
});

function playVocalization(v) {
  const command = v.command || {};
  const f1 = Number(command.f1);
  const f2 = Number(command.f2);
  if (!Number.isFinite(f1) || !Number.isFinite(f2)) return;
  const seconds = Math.min(1, Math.max(0.1, Number(v.durationMs) / 1000 || 0.35));

  const status = document.getElementById('voiceStatus');
  if (status) {
    const what = { imitation: 'Repeating what it heard', naming: 'Saying what this reminds it of', babble: 'Babbling', call: 'Calling — nobody has talked to it for a while' }[v.source] || 'Vocalizing';
    status.textContent = `${what} — ${command.onset === 'nasal' ? '“m” + ' : command.onset === 'stop' ? '“p” + ' : ''}vowel F1 ${f1.toFixed(0)} Hz · F2 ${f2.toFixed(0)} Hz`;
  }
  if (v.source === 'imitation') {
    addLog('info', `🗣️ Repeats a sound it heard (F1 ${f1.toFixed(0)}, F2 ${f2.toFixed(0)} Hz)`);
  } else if (v.source === 'naming') {
    addLog('info', `🗣️ Says the sound that goes with what it perceives (F1 ${f1.toFixed(0)}, F2 ${f2.toFixed(0)} Hz)`);
  }
  if (!voiceEnabled || !voiceCtx) return;

  // The onset first: a nasal murmur (lips closed, sound through the nose) or
  // a stop burst (lips released), then the vowel.
  const now0 = voiceCtx.currentTime;
  let lead = 0;
  if (command.onset === 'nasal') {
    lead = 0.09;
    const hum = voiceCtx.createOscillator();
    hum.type = 'triangle';
    hum.frequency.value = VOICE_PITCH_HZ * 1.04;
    const nasal = voiceCtx.createBiquadFilter();
    nasal.type = 'lowpass';
    nasal.frequency.value = 400;
    const humGain = voiceCtx.createGain();
    humGain.gain.setValueAtTime(0, now0);
    humGain.gain.linearRampToValueAtTime(0.18, now0 + 0.02);
    humGain.gain.linearRampToValueAtTime(0.05, now0 + lead);
    hum.connect(nasal); nasal.connect(humGain); humGain.connect(voiceCtx.destination);
    hum.start(now0); hum.stop(now0 + lead + 0.01);
  } else if (command.onset === 'stop') {
    lead = 0.08;
    const length = Math.floor(voiceCtx.sampleRate * 0.02);
    const buffer = voiceCtx.createBuffer(1, length, voiceCtx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / length);
    const burst = voiceCtx.createBufferSource();
    burst.buffer = buffer;
    const burstGain = voiceCtx.createGain();
    burstGain.gain.value = 0.2;
    burst.connect(burstGain); burstGain.connect(voiceCtx.destination);
    burst.start(now0 + lead - 0.02);
  }
  const now = now0 + lead;
  const source = voiceCtx.createOscillator();
  source.type = 'sawtooth'; // rich in harmonics, like the glottal source
  // The pitch contour of a cry-free vocalization: a little above the usual
  // pitch at the start, settling, then falling toward the end.
  source.frequency.setValueAtTime(VOICE_PITCH_HZ * 1.04, now);
  source.frequency.exponentialRampToValueAtTime(VOICE_PITCH_HZ, now + seconds * 0.4);
  source.frequency.exponentialRampToValueAtTime(VOICE_PITCH_HZ * 0.93, now + seconds);
  // Vibrato: a slow, small wobble of the pitch (a fixed pitch is a machine's).
  const vibrato = voiceCtx.createOscillator();
  vibrato.frequency.value = 5.5;
  const vibratoDepth = voiceCtx.createGain();
  vibratoDepth.gain.value = VOICE_PITCH_HZ * 0.012;
  vibrato.connect(vibratoDepth);
  vibratoDepth.connect(source.frequency);
  // Aspiration: a breath of noise through the same tract.
  const breathLength = Math.floor(voiceCtx.sampleRate * (seconds + 0.05));
  const breathBuffer = voiceCtx.createBuffer(1, breathLength, voiceCtx.sampleRate);
  const breathData = breathBuffer.getChannelData(0);
  for (let i = 0; i < breathLength; i++) breathData[i] = Math.random() * 2 - 1;
  const breath = voiceCtx.createBufferSource();
  breath.buffer = breathBuffer;
  const breathGain = voiceCtx.createGain();
  breathGain.gain.value = 0.015;
  breath.connect(breathGain);

  const out = voiceCtx.createGain();
  const level = 0.3 * Math.min(1, Math.max(0, Number(command.amplitude) || 0.9));
  out.gain.setValueAtTime(0, now);
  out.gain.linearRampToValueAtTime(level, now + 0.04);
  out.gain.setValueAtTime(level, now + seconds - 0.1);
  out.gain.linearRampToValueAtTime(0, now + seconds);

  // The tract: the two formants the brain chose, as wide as in its own
  // model, plus a fixed, weak third one. Two stages per formant: the skirts
  // fall twice as fast, so the harmonics away from the formants (what made
  // it whistle) stay down and the vowel is what is left.
  for (const [freq, bandwidth, gain] of [[f1, FORMANT_BANDWIDTH_HZ.f1, 1.0], [f2, FORMANT_BANDWIDTH_HZ.f2, 0.7], [F3_HZ, FORMANT_BANDWIDTH_HZ.f3, 0.12]]) {
    const first = voiceCtx.createBiquadFilter();
    first.type = 'bandpass';
    first.frequency.value = freq;
    first.Q.value = freq / bandwidth;
    const second = voiceCtx.createBiquadFilter();
    second.type = 'bandpass';
    second.frequency.value = freq;
    second.Q.value = freq / bandwidth;
    const formantGain = voiceCtx.createGain();
    formantGain.gain.value = gain * 2.2; // two stages lose level
    source.connect(first);
    breathGain.connect(first);
    first.connect(second);
    second.connect(formantGain);
    formantGain.connect(out);
  }
  // Lip radiation and the softness of a small tract: nothing sharp above 4 kHz.
  const tilt = voiceCtx.createBiquadFilter();
  tilt.type = 'lowpass';
  tilt.frequency.value = 4000;
  out.connect(tilt);
  tilt.connect(voiceCtx.destination);
  vibrato.start(now);
  breath.start(now);
  source.start(now);
  source.stop(now + seconds + 0.02);
  vibrato.stop(now + seconds + 0.02);
  breath.stop(now + seconds + 0.02);
  source.onended = () => { out.disconnect(); tilt.disconnect(); };

  ownVoiceUntil = performance.now() + seconds * 1000 + 300;
}

// ================================================================
// TEACH — lessons, tests and practice
// ================================================================

/** The whiteboard drawing as the brain sees it: grey levels for the shape, colour apart. */
function currentDrawing() {
  const { pixels, rgb } = pixelsOf(drawCtx);
  const image = { pixels, rgb, width: 64, height: 64 };
  if (drawStrokes.length > 0) image.strokes = drawStrokes.map((s) => ({ points: s.points, durationMs: s.durationMs }));
  return image;
}

let lessonCurve = [];

document.getElementById('teachBtn')?.addEventListener('click', () => {
  const lesson = { repetitions: Math.min(10, Math.max(1, Number(document.getElementById('lessonReps').value) || 4)) };
  if (document.getElementById('lessonUseDrawing').checked) {
    lesson.image = currentDrawing();
  }
  const text = document.getElementById('lessonText').value.trim();
  if (text) lesson.text = text;
  const vowel = document.getElementById('lessonVowel').value;
  if (vowel) lesson.vowel = vowel;

  const parts = [lesson.image ? 'the drawing' : null, lesson.text ? `“${lesson.text}”` : null, lesson.vowel ? `/${lesson.vowel}/` : null].filter(Boolean);
  if (parts.length < 2) {
    document.getElementById('lessonProgress').textContent = 'A lesson pairs at least two things: the drawing, a name, a sound.';
    return;
  }
  lessonCurve = [];
  document.getElementById('lessonProgress').textContent = `Teaching ${parts.join(' + ')} × ${lesson.repetitions}…`;
  addLog('input', `🎓 Lesson: ${parts.join(' + ')} × ${lesson.repetitions}`);
  sendControl('lesson', lesson, 'lesson');
});

document.getElementById('testBtn')?.addEventListener('click', () => {
  addLog('input', '🔍 Test: showing the drawing alone');
  document.getElementById('lessonProgress').textContent = 'Test: showing the drawing alone — watch Perception → Recalls, the voice and the hand.';
  sendControl('input:image', currentDrawing(), 'input/image');
});

function showLessonProgress(d) {
  const box = document.getElementById('lessonProgress');
  if (!box) return;
  if (d.done) {
    const last = lessonCurve[lessonCurve.length - 1];
    box.innerHTML = `${escapeHtml(d.label)} — done. ` +
      (last !== undefined ? `Last recall before repeating: ${Math.round(last * 100)}%${last >= 0.4 ? ' — confident' : ' — needs more repetitions'}.` : '') +
      ` <span class="percept-total">${Number(d.bindings)} shared experiences</span>` + curveHtml();
    addLog('info', `🎓 Lesson done: ${d.label}`);
    return;
  }
  lessonCurve.push(Number(d.confidence) || 0);
  const recalled = d.recalled
    ? [d.recalled.words && d.recalled.words.length ? `“${d.recalled.words.join(' ')}”` : null, d.recalled.visual, d.recalled.auditory].filter(Boolean).map(escapeHtml).join(' + ')
    : '';
  box.innerHTML = `${escapeHtml(d.label)} — repetition ${d.repetition}/${d.of}: ` +
    (d.repetition === 1 ? 'first time, nothing to recall yet' : `recalled ${recalled || 'something'} at ${Math.round(d.confidence * 100)}%${d.confident ? ' ✓' : ''}`) +
    curveHtml();
}

function curveHtml() {
  return `<div class="teach-curve" title="recall confidence before each repetition">` +
    lessonCurve.map((c) => `<div class="teach-bar" style="height:${Math.max(2, Math.round(c * 28))}px" title="${Math.round(c * 100)}%"></div>`).join('') +
    `</div>`;
}

document.getElementById('practiceVoice')?.addEventListener('click', () => {
  document.getElementById('practiceStatus').textContent = 'babbling ×100…';
  addLog('input', '🗣️ Practice: 100 babbles');
  sendControl('practice', { voice: 100 }, 'practice');
});
document.getElementById('practiceHand')?.addEventListener('click', () => {
  document.getElementById('practiceStatus').textContent = 'scribbling ×100…';
  addLog('input', '✍️ Practice: 100 scribbles');
  sendControl('practice', { hand: 100 }, 'practice');
});

function showPracticeProgress(d) {
  const status = document.getElementById('practiceStatus');
  if (!status) return;
  if (d.done) {
    status.textContent = `done — ${Number(d.voice)} babbles, ${Number(d.hand)} scribbles in total`;
    addLog('info', `🎓 Practice done (${Number(d.voice)} babbles, ${Number(d.hand)} scribbles so far)`);
  } else {
    status.textContent = `${d.kind === 'voice' ? 'babbling' : 'scribbling'} ${Number(d.done)}/${Number(d.of)}…`;
  }
}

// ================================================================
// HAND — what the brain draws and writes, and the teacher's feedback
// ================================================================

let handEnabled = false;

function sendControl(type, data, fallbackPath) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type, data }));
  } else {
    fetch(`${API_URL}/${fallbackPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).catch(err => addLog('error', err.message));
  }
}

document.getElementById('toggleHand')?.addEventListener('click', () => {
  const btn = document.getElementById('toggleHand');
  handEnabled = !handEnabled;
  btn.textContent = handEnabled ? '⏹ Disable' : '▶ Enable';
  btn.setAttribute('aria-pressed', String(handEnabled));
  document.getElementById('handStatus').textContent = handEnabled ? 'Scribbling — it learns what marks its commands leave' : 'Still';
  addLog('info', handEnabled ? '✍️ Hand enabled: scribbling + drawing' : '✍️ Hand disabled');
  sendControl('hand', { scribble: handEnabled, copy: handEnabled }, 'hand');
});

document.getElementById('feedbackYes')?.addEventListener('click', () => {
  addLog('info', '👍 Yes, that was right');
  sendControl('feedback', { positive: true }, 'feedback');
});
document.getElementById('feedbackNo')?.addEventListener('click', () => {
  addLog('info', '👎 No, that was not it');
  sendControl('feedback', { positive: false }, 'feedback');
});

/**
 * Something the brain imagined (awake: a daydream; asleep: a dream). Shown
 * in the stream of consciousness with its origin marked — it is not a
 * percept — and in the mind's eye canvas.
 */
function showImagination(im) {
  const box = document.getElementById('thoughtBox');
  const words = Array.isArray(im.words) ? im.words.map(String) : [];
  const evoked = [im.visual, im.colour, im.auditory].filter((x) => typeof x === 'string');
  const sources = (Array.isArray(im.sources) ? im.sources : []).map((s) => String(s).replace(/^[a-z]+:/, ''));
  const dream = im.origin === 'dream';
  if (box) {
    const empty = box.querySelector('.thought-empty');
    if (empty) empty.remove();
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = 'thought-entry thought-imagined';
    entry.innerHTML =
      `<span class="thought-time">${time}</span>` +
      `<span class="thought-emoji">${dream ? '🌙' : '💭'}</span>` +
      `<span class="thought-emotion">${dream ? 'dreamed' : 'imagined'}</span>` +
      `<span class="thought-words">${escapeHtml([...words, ...evoked].join(' · ') || '…')}</span>` +
      `<span class="thought-sources">${escapeHtml(sources.join(' + '))}${Number(im.novelty) >= 0.5 ? ' · never seen together' : ''}</span>`;
    box.insertBefore(entry, box.firstChild);
    while (box.children.length > 30) box.removeChild(box.lastChild);
  }
  const canvas = document.getElementById('mindsEye');
  const side = Number(im.imageSide);
  if (canvas && Array.isArray(im.image) && side > 0 && side <= 64 && im.image.length === side * side) {
    const ctx = canvas.getContext('2d');
    const cell = canvas.width / side;
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < im.image.length; i++) {
      const v = Math.max(0, Math.min(1, Number(im.image[i]) || 0));
      if (v <= 0.05) continue;
      ctx.fillStyle = dream ? `rgba(196, 181, 253, ${v})` : `rgba(248, 250, 252, ${v})`;
      ctx.fillRect((i % side) * cell, Math.floor(i / side) * cell, cell, cell);
    }
    const label = document.getElementById('mindsEyeLabel');
    if (label) label.textContent = dream ? 'dreamed' : 'imagined';
  }
  if (dream || Number(im.novelty) >= 0.5) addLog('info', `${dream ? '🌙 Dreamed' : '💭 Imagined'}: ${escapeHtml([...words, ...evoked].join(' · ') || sources.join(' + '))}`);
}

let drawingAnimation = 0;

function showBrainDrawing(d) {
  const canvas = document.getElementById('brainCanvas2d');
  const side = Number(d.gridSide);
  if (!canvas || !Array.isArray(d.cells) || !(side > 0 && side <= 64)) return;
  const ctx = canvas.getContext('2d');
  const cell = canvas.width / side;
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = d.source === 'scribble' ? '#64748b' : '#f8fafc';
  const ink = (index) => {
    const i = Number(index);
    if (!Number.isInteger(i) || i < 0 || i >= side * side) return;
    ctx.fillRect((i % side) * cell, Math.floor(i / side) * cell, cell, cell);
  };
  const strokes = Array.isArray(d.strokes) ? d.strokes.filter((s) => s && Array.isArray(s.cells)) : [];
  if (strokes.length > 0) {
    // It knows a gesture for this drawing: the strokes appear in order, each over its time.
    const token = ++drawingAnimation;
    let at = 0;
    for (const stroke of strokes) {
      const duration = Math.min(3000, Math.max(80, Number(stroke.durationMs) || 200));
      stroke.cells.forEach((index, k) => {
        setTimeout(() => { if (drawingAnimation === token) { ctx.fillStyle = '#f8fafc'; ink(index); } }, at + (duration * k) / stroke.cells.length);
      });
      at += duration + 120;
    }
  } else {
    drawingAnimation++;
    for (const index of d.cells) ink(index);
  }
  const what = { scribble: 'Scribbling', copy: 'Copying what it saw', 'from-memory': 'Drawing what came to mind', imagined: 'Drawing what it imagined' }[d.source] || 'Drawing';
  const status = document.getElementById('handStatus');
  if (status) status.textContent = what;
  if (d.source !== 'scribble') addLog('info', `✍️ ${what} (${d.cells.length} cells${strokes.length > 0 ? `, ${strokes.length} strokes as it was shown` : ''})`);
}

function showBrainWriting(w) {
  const area = document.getElementById('brainWriting');
  if (!area || typeof w.text !== 'string') return;
  // A textarea's value is plain text: nothing the brain writes can become markup.
  area.value = (area.value ? area.value + ' ' : '') + w.text.slice(0, 40);
  if (area.value.length > 400) area.value = area.value.slice(-400);
  area.scrollTop = area.scrollHeight;
  const cue = String(w.cue);
  addLog('info', cue.startsWith('answer:')
    ? `❓ Answers “${w.text.slice(0, 40)}” (the ${cue.slice(7)} of what is in front)`
    : w.habit === true
      ? `🔁 Out of habit, writes “${w.text.slice(0, 40)}” on seeing ${cue.slice(0, 40)}`
      : `✍️ Writes “${w.text.slice(0, 40)}” on seeing ${cue.slice(0, 40)}`);
}

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
  msgEl.innerHTML = `<div>${escapeHtml(text)}</div><div class="broca-time">${time}</div>`;
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
      const dreams = Array.isArray(result.dreamed) ? result.dreamed.filter(Boolean) : [];
      addLog('info', `💤 Consolidation complete: ${result.memoriesReplayed || 0} memories, ${result.dreams || 0} dreams${result.pruned && result.pruned.length ? `, pruned ${result.pruned.length}` : ''}`);
      for (const d of dreams) addLog('info', `🌙 Dreamed: ${escapeHtml(String(d))}`);
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
window.addEventListener('orientationchange', () => setTimeout(resizeBrainCanvas, 150));
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => resizeBrainCanvas()).observe(brainCanvas.parentElement);
}

// Start (the legend goes in first: the canvas takes the height it leaves)
buildLegend();
resizeBrainCanvas();
drawBrain();
connectWebSocket();

addLog('info', '🧠 Dashboard started');

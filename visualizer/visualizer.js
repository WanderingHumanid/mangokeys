/**
 * MangoKeys Real-Time Neural Transformer Visualizer Engine
 * 60 FPS HTML5 Canvas Renderer for Cross-Attention Rays, Layer Activations,
 * and Live Beam Search Tree Exploration.
 */

(function () {
  'use strict';

  // --- State Management ---
  const state = {
    word: 'namaskaram',
    beamWidth: 4,
    selectedHead: -1, // -1 means All Heads (Average)
    mode: 'live', // 'live' or 'step'
    currentStep: 0,
    totalSteps: 0,
    isPlaying: false,
    playSpeed: 1.0,
    playbackTimer: null,
    telemetryData: null,
    frames: [],
    particles: [],
    hoveredRay: null,
    hoveredNode: null,
    wsConnected: false,
    fps: 60,
    lastFrameTime: performance.now(),
    frameCount: 0,
  };

  // --- DOM Elements ---
  const el = {
    textInput: document.getElementById('textInput'),
    clearInputBtn: document.getElementById('clearInputBtn'),
    beamSelect: document.getElementById('beamSelect'),
    headSelector: document.getElementById('headSelector'),
    sampleChips: document.getElementById('sampleChips'),
    modeLiveBtn: document.getElementById('modeLiveBtn'),
    modeStepBtn: document.getElementById('modeStepBtn'),
    prevStepBtn: document.getElementById('prevStepBtn'),
    playPauseBtn: document.getElementById('playPauseBtn'),
    nextStepBtn: document.getElementById('nextStepBtn'),
    stepSlider: document.getElementById('stepSlider'),
    stepInfo: document.getElementById('stepInfo'),
    speedSelect: document.getElementById('speedSelect'),
    canvasContainer: document.getElementById('canvasContainer'),
    attentionCanvas: document.getElementById('attentionCanvas'),
    attnTooltip: document.getElementById('attnTooltip'),
    beamTreeCanvas: document.getElementById('beamTreeCanvas'),
    softmaxBars: document.getElementById('softmaxBars'),
    encMeters: document.getElementById('encMeters'),
    decMeters: document.getElementById('decMeters'),
    fpsDisplay: document.getElementById('fpsDisplay'),
    latencyDisplay: document.getElementById('latencyDisplay'),
  };

  const attnCtx = el.attentionCanvas.getContext('2d');
  const treeCtx = el.beamTreeCanvas.getContext('2d');

  // --- Resize Canvas for High-DPI Retina Displays ---
  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;

    // Attention Canvas
    const w1 = el.canvasContainer.clientWidth;
    const h1 = el.canvasContainer.clientHeight;
    el.attentionCanvas.width = w1 * dpr;
    el.attentionCanvas.height = h1 * dpr;
    attnCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Tree Canvas
    const treeContainer = document.getElementById('treeContainer');
    const w2 = treeContainer.clientWidth;
    const h2 = treeContainer.clientHeight;
    el.beamTreeCanvas.width = w2 * dpr;
    el.beamTreeCanvas.height = h2 * dpr;
    treeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  window.addEventListener('resize', resizeCanvases);

  // --- WebSocket & API Client ---
  let ws = null;
  let debounceTimeout = null;

  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:5000';
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        state.wsConnected = true;
        updateConnectionStatus(true);
        sendQuery(el.textInput.value);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'telemetry') {
            handleTelemetry(data);
          }
        } catch (e) {
          console.error('[WS] Parse error', e);
        }
      };

      ws.onerror = () => {
        fallbackHttp(el.textInput.value);
      };

      ws.onclose = () => {
        state.wsConnected = false;
        updateConnectionStatus(false);
        setTimeout(initWebSocket, 2500);
      };
    } catch (e) {
      fallbackHttp(el.textInput.value);
    }
  }

  function updateConnectionStatus(connected) {
    const statusEl = document.querySelector('.system-status');
    if (statusEl) {
      if (connected) {
        statusEl.innerHTML = '<span class="status-dot"></span> WEBSOCKET CONNECTED: /ws';
        statusEl.style.color = 'var(--accent-green)';
      } else {
        statusEl.innerHTML = '<span class="status-dot" style="background:var(--accent-amber);box-shadow:0 0 6px var(--accent-amber)"></span> POLLING FALLBACK: /api/extract';
        statusEl.style.color = 'var(--accent-amber)';
      }
    }
  }

  function sendQuery(text) {
    const beam = parseInt(el.beamSelect.value, 10) || 4;
    state.word = text;

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'input', text: text, beam: beam }));
    } else {
      fallbackHttp(text);
    }
  }

  async function fallbackHttp(text) {
    const beam = parseInt(el.beamSelect.value, 10) || 4;
    try {
      const res = await fetch(`/api/extract?text=${encodeURIComponent(text)}&beam=${beam}`);
      if (res.ok) {
        const data = await res.json();
        handleTelemetry(data);
      }
    } catch (err) {
      console.warn('[HTTP] Polling error:', err);
    }
  }

  function debouncedSendQuery(text) {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      sendQuery(text);
    }, 60);
  }

  // --- Telemetry Handler ---
  function handleTelemetry(data) {
    state.telemetryData = data;
    state.frames = data.frames || [];
    state.totalSteps = state.frames.length;

    if (data.latency_ms !== undefined) {
      el.latencyDisplay.textContent = `${data.latency_ms} ms`;
    }

    el.stepSlider.max = Math.max(0, state.totalSteps - 1);

    if (state.mode === 'live') {
      state.currentStep = Math.max(0, state.totalSteps - 1);
      el.stepSlider.value = state.currentStep;
    } else {
      if (state.currentStep >= state.totalSteps) {
        state.currentStep = Math.max(0, state.totalSteps - 1);
        el.stepSlider.value = state.currentStep;
      }
    }

    updateStepUI();
    updateSideMetrics();
    initRayParticles();
  }

  // --- Step Playback Controls ---
  function setMode(newMode) {
    state.mode = newMode;
    if (newMode === 'live') {
      el.modeLiveBtn.classList.add('active');
      el.modeStepBtn.classList.remove('active');
      pausePlayback();
      state.currentStep = Math.max(0, state.totalSteps - 1);
      el.stepSlider.value = state.currentStep;
    } else {
      el.modeStepBtn.classList.add('active');
      el.modeLiveBtn.classList.remove('active');
      state.currentStep = 0;
      el.stepSlider.value = 0;
    }
    updateStepUI();
    updateSideMetrics();
  }

  function stepTo(stepIdx) {
    if (state.totalSteps === 0) return;
    state.currentStep = Math.max(0, Math.min(stepIdx, state.totalSteps - 1));
    el.stepSlider.value = state.currentStep;
    updateStepUI();
    updateSideMetrics();
    initRayParticles();
  }

  function playPlayback() {
    state.isPlaying = true;
    el.playPauseBtn.textContent = '⏸ PAUSE';
    el.playPauseBtn.classList.add('playing');

    clearInterval(state.playbackTimer);
    const interval = 800 / state.playSpeed;
    state.playbackTimer = setInterval(() => {
      if (state.currentStep < state.totalSteps - 1) {
        stepTo(state.currentStep + 1);
      } else {
        stepTo(0);
      }
    }, interval);
  }

  function pausePlayback() {
    state.isPlaying = false;
    el.playPauseBtn.textContent = '▶ PLAY';
    el.playPauseBtn.classList.remove('playing');
    clearInterval(state.playbackTimer);
    state.playbackTimer = null;
  }

  function togglePlayPause() {
    if (state.isPlaying) {
      pausePlayback();
    } else {
      setMode('step');
      playPlayback();
    }
  }

  function updateStepUI() {
    if (state.totalSteps === 0) {
      el.stepInfo.textContent = 'STEP 0 / 0';
      return;
    }

    const curFrame = state.frames[state.currentStep];
    const outChar = curFrame && curFrame.output_tokens.length > 0
      ? curFrame.output_tokens[curFrame.output_tokens.length - 1]
      : '...';

    el.stepInfo.textContent = `STEP ${state.currentStep + 1} / ${state.totalSteps} (${outChar})`;
    el.prevStepBtn.disabled = (state.currentStep === 0);
    el.nextStepBtn.disabled = (state.currentStep >= state.totalSteps - 1);
  }

  // --- Side Metrics: Softmax Bar Chart & Layer Activations ---
  function updateSideMetrics() {
    if (!state.frames || state.frames.length === 0 || !state.frames[state.currentStep]) {
      el.softmaxBars.innerHTML = '<span style="color:var(--text-dim);font-size:0.75rem;">Waiting for token logits...</span>';
      return;
    }

    const frame = state.frames[state.currentStep];

    // 1. Softmax Bars
    const candidates = frame.top_k_candidates || [];
    let barsHtml = '';
    const maxProb = candidates.length > 0 ? Math.max(...candidates.map(c => c.prob)) : 1.0;

    candidates.forEach((cand, idx) => {
      const pct = Math.round(cand.prob * 100);
      const widthPct = Math.max(4, Math.round((cand.prob / (maxProb || 1)) * 100));
      barsHtml += `
        <div class="softmax-bar-row">
          <span class="bar-char">${cand.char}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${widthPct}%;"></div>
          </div>
          <span class="bar-pct">${pct}% (${cand.prob.toFixed(3)})</span>
        </div>
      `;
    });
    el.softmaxBars.innerHTML = barsHtml || '<span style="color:var(--text-dim);">No candidates</span>';

    // 2. Encoder Layer Activations
    const encActs = (frame.layer_activations && frame.layer_activations.encoder) || [0.5, 0.5, 0.5, 0.5];
    let encHtml = '';
    encActs.forEach((act, idx) => {
      const pct = Math.round(act * 100);
      encHtml += `
        <div class="meter-item">
          <span class="meter-name">E${idx + 1}</span>
          <div class="meter-track">
            <div class="meter-fill enc" style="width:${pct}%;"></div>
          </div>
          <span class="meter-val">${pct}%</span>
        </div>
      `;
    });
    el.encMeters.innerHTML = encHtml;

    // 3. Decoder Layer Activations
    const decActs = (frame.layer_activations && frame.layer_activations.decoder) || [0.5, 0.5, 0.5, 0.5];
    let decHtml = '';
    decActs.forEach((act, idx) => {
      const pct = Math.round(act * 100);
      decHtml += `
        <div class="meter-item">
          <span class="meter-name">D${idx + 1}</span>
          <div class="meter-track">
            <div class="meter-fill dec" style="width:${pct}%;"></div>
          </div>
          <span class="meter-val">${pct}%</span>
        </div>
      `;
    });
    el.decMeters.innerHTML = decHtml;
  }

  // --- Dynamic Particle System on Attention Rays ---
  class RayParticle {
    constructor(p0, p1, p2, p3, speed, color) {
      this.p0 = p0;
      this.p1 = p1;
      this.p2 = p2;
      this.p3 = p3;
      this.t = Math.random();
      this.speed = speed;
      this.color = color;
      this.size = Math.random() * 2 + 1.5;
    }

    update() {
      this.t += this.speed;
      if (this.t > 1) {
        this.t = 0;
      }
    }

    draw(ctx) {
      const t = this.t;
      // Cubic Bezier interpolation
      const cx = 3 * (this.p1.x - this.p0.x);
      const bx = 3 * (this.p2.x - this.p1.x) - cx;
      const ax = this.p3.x - this.p0.x - cx - bx;

      const cy = 3 * (this.p1.y - this.p0.y);
      const by = 3 * (this.p2.y - this.p1.y) - cy;
      const ay = this.p3.y - this.p0.y - cy - by;

      const x = ax * t * t * t + bx * t * t + cx * t + this.p0.x;
      const y = ay * t * t * t + by * t * t + cy * t + this.p0.y;

      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.shadowColor = this.color;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.restore();
    }
  }

  function initRayParticles() {
    state.particles = [];
  }

  // --- Attention Canvas Rendering (60 FPS) ---
  function renderAttentionWeb(timestamp) {
    const width = el.canvasContainer.clientWidth;
    const height = el.canvasContainer.clientHeight;
    attnCtx.clearRect(0, 0, width, height);

    const frame = state.frames[state.currentStep];
    if (!frame) {
      drawEmptyCanvasState(attnCtx, width, height);
      return;
    }

    const srcTokens = frame.input_tokens || [];
    const tgtTokens = frame.output_tokens || [];
    const crossAttn = frame.cross_attention || []; // [4, tgt_len, src_len]

    const padX = 60;
    const topY = 56;
    const midY = height / 2;
    const botY = height - 56;

    // Calculate token positions
    const srcCoords = [];
    const srcSpacing = srcTokens.length > 1 ? (width - padX * 2) / (srcTokens.length - 1) : 0;
    for (let i = 0; i < srcTokens.length; i++) {
      const x = srcTokens.length === 1 ? width / 2 : padX + i * srcSpacing;
      srcCoords.push({ x, y: topY, char: srcTokens[i], idx: i });
    }

    const tgtCoords = [];
    const tgtSpacing = tgtTokens.length > 1 ? (width - padX * 2) / (tgtTokens.length - 1) : 0;
    for (let j = 0; j < tgtTokens.length; j++) {
      const x = tgtTokens.length === 1 ? width / 2 : padX + j * tgtSpacing;
      tgtCoords.push({ x, y: botY, char: tgtTokens[j], idx: j });
    }

    // 1. Draw Middle Transformer Pipeline Blocks
    drawTransformerBlocks(attnCtx, width, midY, frame);

    // 2. Draw Attention Bezier Energy Curves
    const activeRayHover = null;
    const selectedHead = state.selectedHead;

    tgtCoords.forEach((tgt, tIdx) => {
      srcCoords.forEach((src, sIdx) => {
        // Calculate weight
        let weight = 0;
        if (selectedHead === -1) {
          // Average across heads
          let sum = 0;
          let count = 0;
          for (let h = 0; h < crossAttn.length; h++) {
            if (crossAttn[h] && crossAttn[h][tIdx] && crossAttn[h][tIdx][sIdx] !== undefined) {
              sum += crossAttn[h][tIdx][sIdx];
              count++;
            }
          }
          weight = count > 0 ? sum / count : 0;
        } else {
          if (crossAttn[selectedHead] && crossAttn[selectedHead][tIdx] && crossAttn[selectedHead][tIdx][sIdx] !== undefined) {
            weight = crossAttn[selectedHead][tIdx][sIdx];
          }
        }

        if (weight > 0.02) {
          const isCurrentTarget = (tIdx === tgtCoords.length - 1);
          drawAttentionRay(attnCtx, src, tgt, midY, weight, isCurrentTarget);

          // Spawn flow particles along strong attention rays
          if (weight > 0.15 && state.particles.length < 80) {
            if (Math.random() < weight * 0.1) {
              const p0 = { x: src.x, y: src.y + 16 };
              const p1 = { x: src.x, y: midY - 30 };
              const p2 = { x: tgt.x, y: midY + 30 };
              const p3 = { x: tgt.x, y: tgt.y - 16 };
              const speed = 0.008 + weight * 0.015;
              const color = isCurrentTarget ? 'rgba(0, 242, 254, 0.95)' : 'rgba(181, 55, 242, 0.85)';
              state.particles.push(new RayParticle(p0, p1, p2, p3, speed, color));
            }
          }
        }
      });
    });

    // 3. Update & Draw Particles
    for (let pIdx = state.particles.length - 1; pIdx >= 0; pIdx--) {
      const p = state.particles[pIdx];
      p.update();
      p.draw(attnCtx);
      if (p.t >= 1) {
        state.particles.splice(pIdx, 1);
      }
    }

    // 4. Draw Source & Target Nodes
    srcCoords.forEach((src) => drawSourceNode(attnCtx, src));
    tgtCoords.forEach((tgt, idx) => {
      const isLatest = (idx === tgtCoords.length - 1);
      drawTargetNode(attnCtx, tgt, isLatest);
    });
  }

  function drawEmptyCanvasState(ctx, w, h) {
    ctx.save();
    ctx.fillStyle = 'rgba(141, 155, 176, 0.3)';
    ctx.font = '14px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('Type in the input box above to visualize neural transliteration tensor flow.', w / 2, h / 2);
    ctx.restore();
  }

  function drawTransformerBlocks(ctx, width, midY, frame) {
    const blockW = 74;
    const blockH = 44;
    const encActs = (frame.layer_activations && frame.layer_activations.encoder) || [0.5, 0.5, 0.5, 0.5];
    const decActs = (frame.layer_activations && frame.layer_activations.decoder) || [0.5, 0.5, 0.5, 0.5];

    // Pipeline Layout: [E1 E2 E3 E4] -> [Latent Memory Bottleneck] -> [D1 D2 D3 D4]
    const gap = 16;
    const totalW = (blockW * 8) + (gap * 8) + 110;
    const startX = (width - totalW) / 2;

    ctx.save();

    // 1. Encoder Blocks (Cyan)
    for (let i = 0; i < 4; i++) {
      const bx = startX + i * (blockW + gap);
      const by = midY - blockH / 2;
      const act = encActs[i] || 0.5;

      ctx.fillStyle = 'rgba(8, 14, 26, 0.85)';
      ctx.strokeStyle = `rgba(0, 242, 254, ${0.2 + act * 0.7})`;
      ctx.lineWidth = 1.5;
      roundRect(ctx, bx, by, blockW, blockH, 6);
      ctx.fill();
      ctx.stroke();

      // Activation bar inside block
      ctx.fillStyle = 'rgba(0, 242, 254, 0.8)';
      ctx.fillRect(bx + 4, by + blockH - 5, (blockW - 8) * act, 2);

      ctx.fillStyle = '#00f2fe';
      ctx.font = '600 11px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`ENC ${i + 1}`, bx + blockW / 2, by + 20);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`L2:${act.toFixed(2)}`, bx + blockW / 2, by + 34);
    }

    // 2. Latent Bottleneck Vector (Glowing Crystal / Pill)
    const memX = startX + 4 * (blockW + gap) + 6;
    const memW = 86;
    const memY = midY - blockH / 2;

    const grad = ctx.createLinearGradient(memX, memY, memX + memW, memY);
    grad.addColorStop(0, '#00f2fe');
    grad.addColorStop(1, '#b537f2');

    ctx.fillStyle = 'rgba(12, 16, 30, 0.95)';
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(0, 242, 254, 0.4)';
    ctx.shadowBlur = 10;
    roundRect(ctx, memX, memY, memW, blockH, 8);
    ctx.fill();
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.fillStyle = '#fff';
    ctx.font = '700 10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('MEMORY', memX + memW / 2, memY + 18);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('d=256', memX + memW / 2, memY + 32);

    // 3. Decoder Blocks (Purple)
    const decStartX = memX + memW + gap + 6;
    for (let j = 0; j < 4; j++) {
      const bx = decStartX + j * (blockW + gap);
      const by = midY - blockH / 2;
      const act = decActs[j] || 0.5;

      ctx.fillStyle = 'rgba(16, 10, 28, 0.85)';
      ctx.strokeStyle = `rgba(181, 55, 242, ${0.2 + act * 0.7})`;
      ctx.lineWidth = 1.5;
      roundRect(ctx, bx, by, blockW, blockH, 6);
      ctx.fill();
      ctx.stroke();

      // Activation bar inside block
      ctx.fillStyle = 'rgba(181, 55, 242, 0.8)';
      ctx.fillRect(bx + 4, by + blockH - 5, (blockW - 8) * act, 2);

      ctx.fillStyle = '#b537f2';
      ctx.font = '600 11px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`DEC ${j + 1}`, bx + blockW / 2, by + 20);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`L2:${act.toFixed(2)}`, bx + blockW / 2, by + 34);
    }

    ctx.restore();
  }

  function drawAttentionRay(ctx, src, tgt, midY, weight, isCurrent) {
    ctx.save();

    const startX = src.x;
    const startY = src.y + 16;
    const endX = tgt.x;
    const endY = tgt.y - 16;

    // Cubic Bezier control points
    const cp1x = startX;
    const cp1y = midY - 20;
    const cp2x = endX;
    const cp2y = midY + 20;

    const lineWidth = Math.max(0.8, weight * 4.5);
    const opacity = Math.min(1.0, 0.15 + weight * 0.85);

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);

    if (isCurrent) {
      ctx.strokeStyle = `rgba(0, 242, 254, ${opacity})`;
      ctx.shadowColor = 'rgba(0, 242, 254, 0.6)';
      ctx.shadowBlur = weight > 0.3 ? 12 : 4;
    } else {
      ctx.strokeStyle = `rgba(181, 55, 242, ${opacity * 0.85})`;
      ctx.shadowColor = 'rgba(181, 55, 242, 0.5)';
      ctx.shadowBlur = weight > 0.3 ? 8 : 2;
    }

    ctx.lineWidth = lineWidth;
    ctx.stroke();

    ctx.restore();
  }

  function drawSourceNode(ctx, node) {
    ctx.save();
    const radius = 17;

    // Outer glow
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(7, 10, 20, 0.95)';
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = 'rgba(0, 242, 254, 0.5)';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.stroke();

    // Text
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#fff';
    ctx.font = '700 13px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.char, node.x, node.y);

    // Index label above
    ctx.fillStyle = 'var(--text-dim)';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText(`s${node.idx}`, node.x, node.y - radius - 5);

    ctx.restore();
  }

  function drawTargetNode(ctx, node, isLatest) {
    ctx.save();
    const radius = 18;

    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 9, 28, 0.95)';

    if (isLatest) {
      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(0, 242, 254, 0.8)';
      ctx.shadowBlur = 14;
    } else {
      ctx.strokeStyle = '#b537f2';
      ctx.lineWidth = 1.5;
      ctx.shadowColor = 'rgba(181, 55, 242, 0.5)';
      ctx.shadowBlur = 8;
    }

    ctx.fill();
    ctx.stroke();

    // Text
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#fff';
    ctx.font = '700 14px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.char, node.x, node.y + 1);

    // Index label below
    ctx.fillStyle = 'var(--text-dim)';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText(`t${node.idx}`, node.x, node.y + radius + 11);

    ctx.restore();
  }

  // --- Live Beam Search Tree Rendering ---
  function renderBeamTree() {
    const width = el.beamTreeCanvas.clientWidth;
    const height = el.beamTreeCanvas.clientHeight;
    treeCtx.clearRect(0, 0, width, height);

    const frame = state.frames[state.currentStep];
    if (!frame || !frame.beam_tree || frame.beam_tree.length === 0) {
      treeCtx.save();
      treeCtx.fillStyle = 'rgba(141, 155, 176, 0.3)';
      treeCtx.font = '12px JetBrains Mono';
      treeCtx.textAlign = 'center';
      treeCtx.fillText('No beam tree active.', width / 2, height / 2);
      treeCtx.restore();
      return;
    }

    const nodes = frame.beam_tree;

    // Group nodes by step
    const stepsMap = {};
    let maxStep = 0;
    nodes.forEach(n => {
      const s = n.step;
      maxStep = Math.max(maxStep, s);
      if (!stepsMap[s]) stepsMap[s] = [];
      stepsMap[s].push(n);
    });

    const padLeft = 40;
    const padRight = 40;
    const padTop = 30;
    const padBottom = 30;

    const stepSpacing = maxStep > 0 ? (width - padLeft - padRight) / Math.max(maxStep, 1) : 0;
    const nodeCoords = {};

    // Compute coordinates for each node
    Object.keys(stepsMap).forEach(sStr => {
      const stepInt = parseInt(sStr, 10);
      const levelNodes = stepsMap[sStr];
      const x = padLeft + stepInt * stepSpacing;
      const count = levelNodes.length;

      levelNodes.forEach((node, idx) => {
        const y = padTop + ((idx + 1) / (count + 1)) * (height - padTop - padBottom);
        nodeCoords[node.id] = { x, y, node };
      });
    });

    // 1. Draw Connecting Tree Edges
    treeCtx.save();
    nodes.forEach(n => {
      if (n.parent && nodeCoords[n.parent] && nodeCoords[n.id]) {
        const p1 = nodeCoords[n.parent];
        const p2 = nodeCoords[n.id];

        treeCtx.beginPath();
        treeCtx.moveTo(p1.x, p1.y);
        treeCtx.bezierCurveTo((p1.x + p2.x) / 2, p1.y, (p1.x + p2.x) / 2, p2.y, p2.x, p2.y);

        if (n.pruned) {
          treeCtx.strokeStyle = 'rgba(255, 158, 0, 0.25)';
          treeCtx.setLineDash([3, 3]);
          treeCtx.lineWidth = 1;
        } else {
          treeCtx.strokeStyle = 'rgba(0, 242, 254, 0.65)';
          treeCtx.setLineDash([]);
          treeCtx.lineWidth = 1.8;
          treeCtx.shadowColor = 'rgba(0, 242, 254, 0.4)';
          treeCtx.shadowBlur = 6;
        }

        treeCtx.stroke();
      }
    });
    treeCtx.restore();

    // 2. Draw Tree Nodes
    treeCtx.save();
    Object.values(nodeCoords).forEach(({ x, y, node }) => {
      const radius = node.pruned ? 9 : 13;

      treeCtx.beginPath();
      treeCtx.arc(x, y, radius, 0, Math.PI * 2);

      if (node.pruned) {
        treeCtx.fillStyle = 'rgba(30, 16, 12, 0.9)';
        treeCtx.strokeStyle = 'rgba(255, 158, 0, 0.7)';
        treeCtx.lineWidth = 1;
        treeCtx.shadowBlur = 0;
      } else {
        treeCtx.fillStyle = 'rgba(8, 22, 34, 0.95)';
        treeCtx.strokeStyle = '#00f2fe';
        treeCtx.lineWidth = 1.5;
        treeCtx.shadowColor = 'rgba(0, 242, 254, 0.6)';
        treeCtx.shadowBlur = 8;
      }

      treeCtx.fill();
      treeCtx.stroke();

      // Node Character
      treeCtx.shadowBlur = 0;
      treeCtx.fillStyle = node.pruned ? '#ff9e00' : '#fff';
      treeCtx.font = node.pruned ? '9px JetBrains Mono' : '700 11px Outfit, sans-serif';
      treeCtx.textAlign = 'center';
      treeCtx.textBaseline = 'middle';
      const label = node.token === '<bos>' ? 'BOS' : node.token;
      treeCtx.fillText(label, x, y + 1);

      // Score Label
      if (!node.pruned && node.step > 0) {
        treeCtx.fillStyle = 'rgba(0, 242, 254, 0.7)';
        treeCtx.font = '8px JetBrains Mono';
        treeCtx.fillText(`${node.score.toFixed(2)}`, x, y + radius + 9);
      }
    });
    treeCtx.restore();
  }

  // --- Animation Main Loop (60 FPS) ---
  function mainLoop(timestamp) {
    // Calculate FPS
    state.frameCount++;
    if (timestamp - state.lastFrameTime >= 1000) {
      state.fps = state.frameCount;
      state.frameCount = 0;
      state.lastFrameTime = timestamp;
      el.fpsDisplay.textContent = `${state.fps} FPS`;
    }

    renderAttentionWeb(timestamp);
    renderBeamTree();

    requestAnimationFrame(mainLoop);
  }

  // --- Helper: Round Rectangle ---
  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  // --- Event Listeners Setup ---
  function setupEventListeners() {
    // 1. Text Input
    el.textInput.addEventListener('input', (e) => {
      // Clear active sample chip
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      debouncedSendQuery(e.target.value);
    });

    el.clearInputBtn.addEventListener('click', () => {
      el.textInput.value = '';
      el.textInput.focus();
      sendQuery('');
    });

    // 2. Beam Select
    el.beamSelect.addEventListener('change', () => {
      sendQuery(el.textInput.value);
    });

    // 3. Head Selector
    el.headSelector.addEventListener('click', (e) => {
      const btn = e.target.closest('.head-pill');
      if (!btn) return;
      document.querySelectorAll('.head-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      state.selectedHead = parseInt(btn.dataset.head, 10);
    });

    // 4. Sample Chips
    el.sampleChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (!chip) return;
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const word = chip.dataset.word;
      el.textInput.value = word;
      sendQuery(word);
    });

    // 5. Dual Playback Mode
    el.modeLiveBtn.addEventListener('click', () => setMode('live'));
    el.modeStepBtn.addEventListener('click', () => setMode('step'));

    // 6. Stepper Controls
    el.prevStepBtn.addEventListener('click', () => {
      setMode('step');
      pausePlayback();
      stepTo(state.currentStep - 1);
    });

    el.nextStepBtn.addEventListener('click', () => {
      setMode('step');
      pausePlayback();
      stepTo(state.currentStep + 1);
    });

    el.playPauseBtn.addEventListener('click', togglePlayPause);

    el.stepSlider.addEventListener('input', (e) => {
      setMode('step');
      pausePlayback();
      stepTo(parseInt(e.target.value, 10));
    });

    el.speedSelect.addEventListener('change', (e) => {
      state.playSpeed = parseFloat(e.target.value) || 1.0;
      if (state.isPlaying) {
        playPlayback(); // restart timer with new interval
      }
    });

    // 7. Mouse Hover Tooltip on Canvas
    el.attentionCanvas.addEventListener('mousemove', (e) => {
      const rect = el.attentionCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const frame = state.frames[state.currentStep];
      if (!frame) return;

      const srcTokens = frame.input_tokens || [];
      const tgtTokens = frame.output_tokens || [];
      const crossAttn = frame.cross_attention || [];

      const padX = 60;
      const topY = 56;
      const botY = el.attentionCanvas.clientHeight - 56;

      let found = null;

      // Check hover on source tokens
      const srcSpacing = srcTokens.length > 1 ? (el.attentionCanvas.clientWidth - padX * 2) / (srcTokens.length - 1) : 0;
      for (let i = 0; i < srcTokens.length; i++) {
        const x = srcTokens.length === 1 ? el.attentionCanvas.clientWidth / 2 : padX + i * srcSpacing;
        const dist = Math.hypot(mouseX - x, mouseY - topY);
        if (dist < 18) {
          found = `Source Token [${i}]: '${srcTokens[i]}'`;
          break;
        }
      }

      // Check hover on target tokens
      if (!found) {
        const tgtSpacing = tgtTokens.length > 1 ? (el.attentionCanvas.clientWidth - padX * 2) / (tgtTokens.length - 1) : 0;
        for (let j = 0; j < tgtTokens.length; j++) {
          const x = tgtTokens.length === 1 ? el.attentionCanvas.clientWidth / 2 : padX + j * tgtSpacing;
          const dist = Math.hypot(mouseX - x, mouseY - botY);
          if (dist < 18) {
            // Find top attended source
            let bestSrcIdx = 0;
            let bestW = 0;
            for (let s = 0; s < srcTokens.length; s++) {
              let w = 0;
              for (let h = 0; h < crossAttn.length; h++) {
                if (crossAttn[h] && crossAttn[h][j]) w += crossAttn[h][j][s] || 0;
              }
              w /= Math.max(crossAttn.length, 1);
              if (w > bestW) {
                bestW = w;
                bestSrcIdx = s;
              }
            }
            const sChar = srcTokens[bestSrcIdx] || '';
            found = `Target Token [${j}]: '${tgtTokens[j]}' ⟵ Highest Attn to '${sChar}' (${(bestW * 100).toFixed(1)}%)`;
            break;
          }
        }
      }

      if (found) {
        el.attnTooltip.style.display = 'block';
        el.attnTooltip.style.left = `${mouseX}px`;
        el.attnTooltip.style.top = `${mouseY}px`;
        el.attnTooltip.innerHTML = found;
      } else {
        el.attnTooltip.style.display = 'none';
      }
    });

    el.attentionCanvas.addEventListener('mouseleave', () => {
      el.attnTooltip.style.display = 'none';
    });
  }

  // --- Bootstrap Engine ---
  window.addEventListener('DOMContentLoaded', () => {
    resizeCanvases();
    setupEventListeners();
    initWebSocket();
    requestAnimationFrame(mainLoop);
  });

})();

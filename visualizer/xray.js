/**
 * MangoKeys Deep Transformer X-Ray & Model Mechanics Engine.
 *
 * Visualizes authentic 256-D hidden states, harmonic sinusoidal positional encodings,
 * multi-head Q-K-V cross-attention, step entropy, temperature sampling, and beam search trees.
 */

(function () {
  'use strict';

  // State
  let currentData = null;
  let activeStepIndex = 0;
  let currentTemperature = 1.0;
  let activeHead = -1; // -1: Average, 0..3: Specific Heads
  let fetchTimeout = null;
  let ws = null;

  // DOM Elements
  const el = {
    textInput: document.getElementById('textInput'),
    clearBtn: document.getElementById('clearBtn'),
    chipsList: document.getElementById('chipsList'),
    inputWarning: document.getElementById('inputWarning'),
    idleBanner: document.getElementById('idleBanner'),
    activeContent: document.getElementById('activeContent'),

    wsStatusText: document.getElementById('wsStatusText'),
    wsPill: document.getElementById('wsPill'),
    latencyBadge: document.getElementById('latencyBadge'),

    heroSrc: document.getElementById('heroSrc'),
    heroTgt: document.getElementById('heroTgt'),
    heroConf: document.getElementById('heroConf'),
    heroSteps: document.getElementById('heroSteps'),

    srcLenBadge: document.getElementById('srcLenBadge'),
    latentCards: document.getElementById('latentCards'),
    peGrid: document.getElementById('peGrid'),

    headSelector: document.getElementById('headSelector'),
    srcTokenStrip: document.getElementById('srcTokenStrip'),
    tgtTokenStrip: document.getElementById('tgtTokenStrip'),
    svgRibbonWrapper: document.getElementById('svgRibbonWrapper'),
    attentionSvg: document.getElementById('attentionSvg'),
    attentionInfoBar: document.getElementById('attentionInfoBar'),

    scrubberStrip: document.getElementById('scrubberStrip'),
    activeStepChar: document.getElementById('activeStepChar'),
    activeStepCategory: document.getElementById('activeStepCategory'),
    activeStepPrefix: document.getElementById('activeStepPrefix'),
    entropyVal: document.getElementById('entropyVal'),
    gaugeFill: document.getElementById('gaugeFill'),
    gaugeStatus: document.getElementById('gaugeStatus'),

    tempSlider: document.getElementById('tempSlider'),
    tempValDisplay: document.getElementById('tempValDisplay'),
    distBarsContainer: document.getElementById('distBarsContainer'),

    beamTreeContainer: document.getElementById('beamTreeContainer'),
    beamTableBody: document.getElementById('beamTableBody'),
  };

  // -------------------------------------------------------------
  // Networking: REST & WebSocket
  // -------------------------------------------------------------

  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:5000';
    const url = `${protocol}//${host}/ws`;

    try {
      ws = new WebSocket(url);

      ws.onopen = () => {
        el.wsStatusText.textContent = 'Connected (Sync)';
        el.wsPill.querySelector('.dot').className = 'dot pulse-green';
      };

      ws.onclose = () => {
        el.wsStatusText.textContent = 'Reconnecting...';
        el.wsPill.querySelector('.dot').className = 'dot';
        setTimeout(initWebSocket, 2000);
      };

      ws.onerror = () => {
        el.wsStatusText.textContent = 'Offline';
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'telemetry' && msg.data) {
            currentData = msg.data;
            renderAll(msg.data);
          } else if (msg.type === 'desktop_typing' && msg.word) {
            el.textInput.value = msg.word;
            fetchTelemetry(msg.word);
          }
        } catch (e) {
          console.error('WS parse error:', e);
        }
      };
    } catch (e) {
      console.warn('WebSocket init failed:', e);
    }
  }

  async function fetchTelemetry(word) {
    const trimmed = (word || '').trim();
    if (!trimmed) {
      showIdleState();
      return;
    }

    try {
      const res = await fetch(`/api/extract?text=${encodeURIComponent(trimmed)}&beam=4`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      currentData = data;
      renderAll(data);
    } catch (err) {
      console.error('Fetch telemetry error:', err);
    }
  }

  function debouncedFetch(word) {
    clearTimeout(fetchTimeout);
    fetchTimeout = setTimeout(() => fetchTelemetry(word), 80);
  }

  function showIdleState() {
    el.idleBanner.style.display = 'flex';
    el.activeContent.style.display = 'none';
    el.latencyBadge.textContent = '0.0 ms (Standby)';
    el.latencyBadge.className = 'pill-value text-green';
  }

  // -------------------------------------------------------------
  // Rendering Orchestrator
  // -------------------------------------------------------------

  function renderAll(data) {
    if (!data || !data.active) {
      showIdleState();
      return;
    }

    el.idleBanner.style.display = 'none';
    el.activeContent.style.display = 'flex';

    if (data.is_malayalam_input) {
      el.inputWarning.textContent = data.message || 'Malayalam input detected.';
      el.inputWarning.style.display = 'block';
    } else {
      el.inputWarning.style.display = 'none';
    }

    el.latencyBadge.textContent = `${data.latency_ms || 0} ms`;

    renderHero(data);
    renderLatentSpace(data);
    renderAttention(data);
    renderScrubber(data);
    renderStepDetails();
    renderBeamTree(data);
  }

  // -------------------------------------------------------------
  // Hero Summary Strip
  // -------------------------------------------------------------

  function renderHero(data) {
    el.heroSrc.textContent = data.word || '';
    el.heroTgt.textContent = data.top_candidate || '';
    const topCand = (data.candidates && data.candidates[0]) ? data.candidates[0] : null;
    el.heroConf.textContent = topCand ? `${topCand.confidence_pct}%` : '98.5%';
    el.heroSteps.textContent = `${(data.steps || []).length} steps`;
  }

  // -------------------------------------------------------------
  // SECTION 1: Latent Vector Sparklines & Positional Waves
  // -------------------------------------------------------------

  function drawSparkline(canvas, values, strokeColor) {
    if (!canvas || !values || !values.length) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    const min = Math.min(...values, -0.5);
    const max = Math.max(...values, 0.5);
    const range = (max - min) || 1.0;

    // Draw zero center line
    const zeroY = h - ((0 - min) / range) * h;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();

    // Draw activation bars / area
    const barW = Math.max(2, w / values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      const y = h - ((v - min) / range) * h;
      const x = i * barW;

      ctx.fillStyle = v >= 0 ? strokeColor : 'rgba(255, 82, 82, 0.7)';
      const barH = Math.abs(y - zeroY);
      ctx.fillRect(x, Math.min(y, zeroY), barW - 1, Math.max(2, barH));
    }
  }

  function renderLatentSpace(data) {
    const srcTokens = data.src_tokens || [];
    const srcNorms = data.src_norms || [];
    const srcVectors = data.src_vectors_sample || [];
    const srcPe = data.src_pe_sample || [];

    el.srcLenBadge.textContent = srcTokens.length;

    // Render 256-D Token Vector Cards
    el.latentCards.innerHTML = '';
    srcTokens.forEach((tok, idx) => {
      const norm = srcNorms[idx] !== undefined ? srcNorms[idx] : 1.0;
      const card = document.createElement('div');
      card.className = 'latent-card';
      card.innerHTML = `
        <div class="latent-card-header">
          <span class="latent-token-char">${escapeHtml(tok)}</span>
          <span class="latent-token-id">pos=${idx} &bull; tok_id=${tok.charCodeAt(0)}</span>
        </div>
        <div class="latent-norm-row">
          <span class="sub-label">L2 VECTOR NORM:</span>
          <span class="latent-norm-val">&Vert;v&Vert; = ${norm}</span>
        </div>
        <div class="sparkline-wrap">
          <span class="sparkline-label">256-D HIDDEN ACTIVATIONS (SLICES 0..31):</span>
          <canvas class="sparkline-canvas" id="sparkline_${idx}"></canvas>
        </div>
      `;
      el.latentCards.appendChild(card);

      setTimeout(() => {
        const cv = document.getElementById(`sparkline_${idx}`);
        if (cv && srcVectors[idx]) {
          drawSparkline(cv, srcVectors[idx], 'rgba(0, 242, 254, 0.85)');
        }
      }, 10);
    });

    // Render Sinusoidal Positional Encoding Wave Harmonics
    el.peGrid.innerHTML = '';
    srcTokens.forEach((tok, idx) => {
      const peValues = srcPe[idx] || [];
      const band = document.createElement('div');
      band.className = 'pe-band';
      band.innerHTML = `
        <div class="pe-band-title">pos = ${idx} &bull; '${escapeHtml(tok)}' harmonic waves</div>
        <canvas class="pe-canvas" id="pe_canvas_${idx}"></canvas>
      `;
      el.peGrid.appendChild(band);

      setTimeout(() => {
        const cv = document.getElementById(`pe_canvas_${idx}`);
        if (cv && peValues.length) {
          drawSparkline(cv, peValues, 'rgba(181, 55, 242, 0.85)');
        }
      }, 10);
    });
  }

  // -------------------------------------------------------------
  // SECTION 2: Multi-Head Q-K-V Cross-Attention
  // -------------------------------------------------------------

  function renderAttention(data) {
    const srcTokens = data.src_tokens || [];
    const tgtTokens = data.tgt_tokens || [];

    // Render source token pills
    el.srcTokenStrip.innerHTML = '';
    srcTokens.forEach((tok, idx) => {
      const pill = document.createElement('div');
      pill.className = 'token-pill src-pill';
      pill.setAttribute('data-idx', idx);
      pill.innerHTML = `
        <span class="token-char">${escapeHtml(tok)}</span>
        <span class="token-pos">s_${idx}</span>
      `;
      el.srcTokenStrip.appendChild(pill);
    });

    // Render target token pills
    el.tgtTokenStrip.innerHTML = '';
    tgtTokens.forEach((tok, idx) => {
      const pill = document.createElement('div');
      pill.className = 'token-pill tgt-pill';
      pill.setAttribute('data-idx', idx);
      pill.innerHTML = `
        <span class="token-char">${escapeHtml(tok)}</span>
        <span class="token-pos">t_${idx}</span>
      `;
      el.tgtTokenStrip.appendChild(pill);
    });

    // Delay drawing ribbons until DOM layout stabilizes
    setTimeout(() => drawAttentionRibbons(data), 30);
  }

  function drawAttentionRibbons(data) {
    if (!data) return;
    const srcTokens = data.src_tokens || [];
    const tgtTokens = data.tgt_tokens || [];

    // Select matrix based on active head
    let matrix = data.attention_matrix || [];
    if (activeHead >= 0 && data.attention_heads && data.attention_heads[activeHead]) {
      matrix = data.attention_heads[activeHead];
    }

    const wrapperRect = el.svgRibbonWrapper.getBoundingClientRect();
    const height = 150;
    el.attentionSvg.setAttribute('viewBox', `0 0 ${wrapperRect.width} ${height}`);

    const srcPills = el.srcTokenStrip.querySelectorAll('.src-pill');
    const tgtPills = el.tgtTokenStrip.querySelectorAll('.tgt-pill');

    if (!srcPills.length || !tgtPills.length) {
      el.attentionSvg.innerHTML = '';
      return;
    }

    const paths = [];

    tgtPills.forEach((tPill, j) => {
      const tRect = tPill.getBoundingClientRect();
      const tx = tRect.left + tRect.width / 2 - wrapperRect.left;
      const ty = height;
      const row = matrix[j] || [];

      srcPills.forEach((sPill, i) => {
        const sRect = sPill.getBoundingClientRect();
        const sx = sRect.left + sRect.width / 2 - wrapperRect.left;
        const sy = 0;

        const weight = row[i] !== undefined ? row[i] : 0.0;
        if (weight < 0.02) return;

        const strokeWidth = Math.max(1, Math.min(14, weight * 16));
        const opacity = Math.min(0.95, Math.max(0.12, weight * 1.2));

        // Cubic Bezier curve
        const cp1x = sx;
        const cp1y = sy + height * 0.45;
        const cp2x = tx;
        const cp2y = ty - height * 0.45;
        const d = `M ${sx} ${sy} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${tx} ${ty}`;

        let strokeColor = '#00f2fe';
        if (activeHead === 0) strokeColor = '#00e676';
        else if (activeHead === 1) strokeColor = '#ff9e00';
        else if (activeHead === 2) strokeColor = '#b537f2';
        else if (activeHead === 3) strokeColor = '#00f2fe';

        paths.push(`
          <path 
            class="ribbon-path" 
            d="${d}" 
            stroke="${strokeColor}" 
            stroke-width="${strokeWidth}" 
            stroke-opacity="${opacity}" 
            fill="none" 
            data-src-idx="${i}" 
            data-tgt-idx="${j}" 
            data-weight="${weight}" 
          />
        `);
      });
    });

    el.attentionSvg.innerHTML = paths.join('');

    // Attach interactive hover inspection
    el.attentionSvg.querySelectorAll('.ribbon-path').forEach(p => {
      p.addEventListener('mouseenter', () => {
        const i = p.getAttribute('data-src-idx');
        const j = p.getAttribute('data-tgt-idx');
        const w = parseFloat(p.getAttribute('data-weight'));
        const sChar = srcTokens[i] || '';
        const tChar = tgtTokens[j] || '';
        const headLabel = activeHead === -1 ? 'Cross-Head Ensemble Average' : `Attention Head ${activeHead + 1}`;
        el.attentionInfoBar.innerHTML = `
          <strong>[${headLabel}]</strong> Focus: Target glyph <strong>'${escapeHtml(tChar)}'</strong> &larr; Source letter <strong>'${escapeHtml(sChar)}'</strong> &bull; Attention Weight: <span style="color:#00f2fe; font-weight:700;">${(w * 100).toFixed(1)}%</span> &bull; (Q&middot;K<sup>T</sup> / &radic;64 scaled score)
        `;
      });
    });
  }

  // Head selector buttons
  el.headSelector.addEventListener('click', (e) => {
    const btn = e.target.closest('.head-btn');
    if (!btn) return;
    el.headSelector.querySelectorAll('.head-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeHead = parseInt(btn.getAttribute('data-head'), 10);
    drawAttentionRibbons(currentData);
  });

  // -------------------------------------------------------------
  // SECTION 3: Step Scrubber, Softmax Spectrum & Temperature
  // -------------------------------------------------------------

  function renderScrubber(data) {
    const steps = data.steps || [];
    el.scrubberStrip.innerHTML = '';

    if (!steps.length) {
      el.activeStepChar.textContent = '—';
      return;
    }

    // Default to first step if activeStepIndex is out of range
    if (activeStepIndex >= steps.length) {
      activeStepIndex = 0;
    }

    steps.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.className = `scrub-step-btn ${idx === activeStepIndex ? 'active' : ''}`;
      btn.setAttribute('data-idx', idx);
      const isEos = s.is_eos;
      btn.innerHTML = `
        <span class="step-num-badge">Step ${s.step}</span>
        <span class="step-char-preview">${isEos ? '&lt;EOS&gt;' : escapeHtml(s.selected_token)}</span>
      `;
      btn.addEventListener('click', () => {
        activeStepIndex = idx;
        el.scrubberStrip.querySelectorAll('.scrub-step-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderStepDetails();
      });
      el.scrubberStrip.appendChild(btn);
    });
  }

  function calculateSoftmaxWithTemperature(candidates, temp) {
    if (!candidates || !candidates.length) return [];
    const t = Math.max(0.05, Math.min(3.0, temp || 1.0));

    // Extract raw logits (or synthetic approximation from prob if missing)
    const logits = candidates.map(c => {
      if (c.raw_logit !== undefined) return c.raw_logit;
      return Math.log(Math.max(1e-6, c.prob || 0.01));
    });

    // Compute exp(logit / T) with numerical stability
    const scaled = logits.map(z => z / t);
    const maxScaled = Math.max(...scaled);
    const exps = scaled.map(s => Math.exp(s - maxScaled));
    const sumExps = exps.reduce((a, b) => a + b, 0);

    return candidates.map((cand, idx) => {
      const newProb = exps[idx] / sumExps;
      return {
        ...cand,
        prob: newProb,
        pct: (newProb * 100).toFixed(1),
      };
    });
  }

  function renderStepDetails() {
    if (!currentData || !currentData.steps || !currentData.steps[activeStepIndex]) {
      return;
    }

    const step = currentData.steps[activeStepIndex];
    const isEos = step.is_eos;
    el.activeStepChar.textContent = isEos ? '<EOS>' : step.selected_token;

    const prefixStr = (step.prefix || []).join('');
    el.activeStepPrefix.textContent = `Prefix: "${prefixStr || '—'}"`;

    // Recalculate candidate distribution based on temperature
    const rawCandidates = step.top_candidates || [];
    const adjustedCandidates = calculateSoftmaxWithTemperature(rawCandidates, currentTemperature);

    const topCand = adjustedCandidates[0];
    el.activeStepCategory.textContent = topCand ? `${topCand.category}` : 'Consonant';

    // Shannon Entropy Calculation
    let entropy = 0.0;
    adjustedCandidates.forEach(c => {
      const p = c.prob;
      if (p > 1e-8) {
        entropy -= p * Math.log2(p);
      }
    });

    el.entropyVal.textContent = `${entropy.toFixed(3)} bits`;

    // Certainty Meter: low entropy = high confidence
    const maxEntropy = 3.0;
    const certaintyPct = Math.max(0, Math.min(100, (1.0 - (entropy / maxEntropy)) * 100));
    el.gaugeFill.style.width = `${certaintyPct}%`;

    if (certaintyPct > 80) {
      el.gaugeStatus.textContent = `● High Certainty (${certaintyPct.toFixed(1)}%)`;
      el.gaugeStatus.className = 'gauge-status text-green';
    } else if (certaintyPct > 50) {
      el.gaugeStatus.textContent = `▲ Moderate Ambiguity (${certaintyPct.toFixed(1)}%)`;
      el.gaugeStatus.className = 'gauge-status text-cyan';
    } else {
      el.gaugeStatus.textContent = `▼ High Branching Uncertainty (${certaintyPct.toFixed(1)}%)`;
      el.gaugeStatus.className = 'gauge-status text-purple';
    }

    // Render Top-8 Candidate Distribution Bars
    el.distBarsContainer.innerHTML = '';
    adjustedCandidates.forEach((cand, idx) => {
      const row = document.createElement('div');
      row.className = 'dist-bar-row';
      const isTop = (idx === 0);
      const barColor = cand.color || '#00f2fe';

      row.innerHTML = `
        <span class="dist-char text-malayalam" style="color: ${isTop ? '#fff' : 'var(--text-secondary)'};">
          ${cand.char === '<EOS>' ? '&loz;' : escapeHtml(cand.char)}
        </span>
        <span class="dist-badge ${cand.badge || 'badge-consonant'}">${escapeHtml(cand.category || '')}</span>
        <div class="dist-track">
          <div class="dist-fill" style="width: ${cand.pct}%; background: ${barColor};"></div>
        </div>
        <span class="dist-pct" style="color: ${isTop ? '#00e676' : 'var(--text-muted)'};">${cand.pct}%</span>
      `;
      el.distBarsContainer.appendChild(row);
    });
  }

  // Temperature Slider Event
  el.tempSlider.addEventListener('input', (e) => {
    currentTemperature = parseFloat(e.target.value);
    el.tempValDisplay.textContent = `T = ${currentTemperature.toFixed(2)}`;
    renderStepDetails();
  });

  // -------------------------------------------------------------
  // SECTION 4: Autoregressive Beam Search Exploration Tree
  // -------------------------------------------------------------

  function renderBeamTree(data) {
    const beamTree = data.beam_tree || [];
    el.beamTreeContainer.innerHTML = '';

    if (!beamTree.length) {
      el.beamTreeContainer.innerHTML = '<div style="color:var(--text-muted); padding:16px;">Beam search data initializing...</div>';
      return;
    }

    beamTree.forEach(stepData => {
      const col = document.createElement('div');
      col.className = 'beam-step-col';
      col.innerHTML = `<div class="beam-step-header">STEP ${stepData.step}</div>`;

      (stepData.nodes || []).forEach((node, nodeIdx) => {
        const nDiv = document.createElement('div');
        nDiv.className = `beam-node ${nodeIdx === 0 ? 'top-beam' : ''}`;
        nDiv.innerHTML = `
          <div class="beam-node-char">
            <span class="beam-char-txt">${node.is_eos ? '&lt;EOS&gt;' : escapeHtml(node.char)}</span>
            <span class="beam-score-badge">&sum;=${node.score}</span>
          </div>
          <div class="beam-node-seq">${escapeHtml(node.text)}</div>
        `;
        col.appendChild(nDiv);
      });

      el.beamTreeContainer.appendChild(col);
    });

    // Populate summary table
    const candidates = data.candidates || [];
    el.beamTableBody.innerHTML = '';
    candidates.forEach((cand, idx) => {
      const tr = document.createElement('tr');
      const isTop = (idx === 0);
      tr.innerHTML = `
        <td><span class="rank-badge ${isTop ? 'rank-1' : ''}">${idx + 1}</span></td>
        <td style="font-family:'Noto Sans Malayalam', sans-serif; font-size:16px; font-weight:700; color:${isTop ? 'var(--accent-cyan)' : '#fff'};">${escapeHtml(cand.candidate)}</td>
        <td style="font-family:'JetBrains Mono', monospace; color:var(--accent-green);">${cand.score}</td>
        <td style="font-family:'JetBrains Mono', monospace;">${cand.confidence_pct}%</td>
        <td><span class="badge-mini ${isTop ? 'cyan-badge' : ''}">${isTop ? 'Selected (Rank 1)' : 'Pruned Branch'}</span></td>
      `;
      el.beamTableBody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // Helpers & Event Listeners
  // -------------------------------------------------------------

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Search input typing
  el.textInput.addEventListener('input', (e) => {
    debouncedFetch(e.target.value);
  });

  // Clear button
  el.clearBtn.addEventListener('click', () => {
    el.textInput.value = '';
    showIdleState();
    el.textInput.focus();
  });

  // Sample Chips
  el.chipsList.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    el.chipsList.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    const word = chip.getAttribute('data-word');
    el.textInput.value = word;
    fetchTelemetry(word);
  });

  // Resize window handler to redraw attention ribbons
  window.addEventListener('resize', () => {
    if (currentData) {
      drawAttentionRibbons(currentData);
    }
  });

  // Initialize
  initWebSocket();
  fetchTelemetry(el.textInput.value || 'ente');

})();

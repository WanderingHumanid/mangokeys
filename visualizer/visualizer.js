/**
 * MangoKeys Real-Time Neural Transformer Visualizer
 * Fast, Clean, Reactive SVG & DOM Engine.
 */

(function () {
  'use strict';

  // State
  let currentData = null;
  let activeHoveredSrc = null;
  let activeHoveredTgt = null;
  let currentView = 'ribbon'; // 'ribbon' or 'matrix'
  let ws = null;
  let desktopFeedItems = [];

  // DOM Elements
  const el = {
    textInput: document.getElementById('textInput'),
    clearBtn: document.getElementById('clearBtn'),
    chipsList: document.getElementById('chipsList'),
    wsPill: document.getElementById('wsPill'),
    wsStatusText: document.getElementById('wsStatusText'),
    engineBadge: document.getElementById('engineBadge'),
    latencyBadge: document.getElementById('latencyBadge'),
    heroSourceWord: document.getElementById('heroSourceWord'),
    heroTargetWord: document.getElementById('heroTargetWord'),
    viewRibbonBtn: document.getElementById('viewRibbonBtn'),
    viewMatrixBtn: document.getElementById('viewMatrixBtn'),
    ribbonStage: document.getElementById('ribbonStage'),
    heatmapStage: document.getElementById('heatmapStage'),
    srcTokenStrip: document.getElementById('srcTokenStrip'),
    tgtTokenStrip: document.getElementById('tgtTokenStrip'),
    attentionSvg: document.getElementById('attentionSvg'),
    svgRibbonWrapper: document.getElementById('svgRibbonWrapper'),
    heatmapGrid: document.getElementById('heatmapGrid'),
    candidatesList: document.getElementById('candidatesList'),
    desktopFeed: document.getElementById('desktopFeed'),
  };

  // --- WebSocket Connection ---
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:5000';
    const url = `${protocol}//${host}/ws`;

    try {
      ws = new WebSocket(url);

      ws.onopen = () => {
        el.wsPill.classList.remove('disconnected');
        el.wsStatusText.textContent = 'Connected';
        // Send initial word
        fetchTelemetry(el.textInput.value || 'ente');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'desktop_sync') {
            handleDesktopEvent(msg);
          } else if (msg.type === 'telemetry') {
            renderData(msg.data);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onclose = () => {
        el.wsPill.classList.add('disconnected');
        el.wsStatusText.textContent = 'Offline (Reconnecting)';
        setTimeout(initWebSocket, 2000);
      };

      ws.onerror = () => {
        el.wsPill.classList.add('disconnected');
        el.wsStatusText.textContent = 'Disconnected';
      };
    } catch (e) {
      console.warn('WebSocket init error, falling back to HTTP:', e);
    }
  }

  // --- Fetch Telemetry via Fast HTTP Fallback ---
  let fetchTimer = null;
  async function fetchTelemetry(word) {
    if (!word) return;
    try {
      const res = await fetch(`/api/extract?text=${encodeURIComponent(word)}`);
      if (res.ok) {
        const data = await res.json();
        renderData(data);
      }
    } catch (e) {
      console.warn('Fetch error:', e);
    }
  }

  function debouncedFetch(word) {
    clearTimeout(fetchTimer);
    fetchTimer = setTimeout(() => fetchTelemetry(word), 60);
  }

  // --- Handle Real-Time Desktop Sync Events ---
  function handleDesktopEvent(event) {
    const word = event.word || '';
    const topCand = event.top_candidate || '';
    const latency = event.latency_ms || 0.0;
    const now = new Date().toLocaleTimeString();

    // Add to desktop feed
    desktopFeedItems.unshift({ word, topCand, latency, now });
    if (desktopFeedItems.length > 8) desktopFeedItems.pop();
    renderDesktopFeed();

    // Update main display if user isn't actively typing in the search box
    if (document.activeElement !== el.textInput) {
      el.textInput.value = word;
      if (event.telemetry) {
        renderData(event.telemetry);
      } else {
        debouncedFetch(word);
      }
    }
  }

  function renderDesktopFeed() {
    if (desktopFeedItems.length === 0) {
      el.desktopFeed.innerHTML = '<div class="feed-empty">Waiting for desktop typing events... (type anywhere on your desktop)</div>';
      return;
    }

    el.desktopFeed.innerHTML = desktopFeedItems.map(item => `
      <div class="feed-item">
        <div class="feed-pair">
          <span class="feed-src">${escapeHtml(item.word)}</span>
          <span class="feed-arr">&rarr;</span>
          <span class="feed-tgt">${escapeHtml(item.topCand)}</span>
        </div>
        <div class="feed-meta">
          <span>⚡ ${item.latency.toFixed(2)} ms</span>
          <span>${item.now}</span>
        </div>
      </div>
    `).join('');
  }

  // --- Render Telemetry Data ---
  function renderData(data) {
    if (!data) return;
    currentData = data;

    // 1. Update Hero
    el.heroSourceWord.textContent = data.word || '';
    el.heroTargetWord.textContent = data.top_candidate || '';
    el.latencyBadge.textContent = `⚡ ${(data.latency_ms || 0.08).toFixed(2)} ms`;
    el.engineBadge.textContent = data.engine === 'onnx' ? 'Neural ONNX (Seq2Seq)' : 'RuleEngine (Native)';

    // 2. Render Tokens & Ribbons
    renderTokens(data.src_tokens || [], data.tgt_tokens || []);
    drawRibbons(data.src_tokens || [], data.tgt_tokens || [], data.attention_matrix || []);

    // 3. Render Matrix Heatmap
    renderHeatmap(data.src_tokens || [], data.tgt_tokens || [], data.attention_matrix || []);

    // 4. Render Candidates
    renderCandidates(data.candidates || []);
  }

  // --- Render Token Strips ---
  function renderTokens(srcTokens, tgtTokens) {
    // Source tokens
    el.srcTokenStrip.innerHTML = srcTokens.map((char, idx) => `
      <div class="token-pill src ${activeHoveredSrc === idx ? 'active' : ''}" data-idx="${idx}">
        ${escapeHtml(char)}
      </div>
    `).join('');

    // Target tokens
    el.tgtTokenStrip.innerHTML = tgtTokens.map((char, idx) => `
      <div class="token-pill tgt ${activeHoveredTgt === idx ? 'active' : ''}" data-idx="${idx}">
        ${escapeHtml(char)}
      </div>
    `).join('');

    // Attach hover listeners
    el.srcTokenStrip.querySelectorAll('.token-pill').forEach(pill => {
      const idx = parseInt(pill.getAttribute('data-idx'), 10);
      pill.addEventListener('mouseenter', () => {
        activeHoveredSrc = idx;
        activeHoveredTgt = null;
        redrawRibbons();
      });
      pill.addEventListener('mouseleave', () => {
        activeHoveredSrc = null;
        redrawRibbons();
      });
    });

    el.tgtTokenStrip.querySelectorAll('.token-pill').forEach(pill => {
      const idx = parseInt(pill.getAttribute('data-idx'), 10);
      pill.addEventListener('mouseenter', () => {
        activeHoveredTgt = idx;
        activeHoveredSrc = null;
        redrawRibbons();
      });
      pill.addEventListener('mouseleave', () => {
        activeHoveredTgt = null;
        redrawRibbons();
      });
    });
  }

  // --- Draw Fast SVG Cross-Attention Ribbons ---
  function drawRibbons(srcTokens, tgtTokens, matrix) {
    if (!srcTokens.length || !tgtTokens.length || !matrix.length) {
      el.attentionSvg.innerHTML = '';
      return;
    }

    const wrapperRect = el.svgRibbonWrapper.getBoundingClientRect();
    const width = wrapperRect.width || 800;
    const height = 140;

    const srcPills = el.srcTokenStrip.querySelectorAll('.token-pill');
    const tgtPills = el.tgtTokenStrip.querySelectorAll('.token-pill');

    if (!srcPills.length || !tgtPills.length) return;

    const paths = [];

    // Map each target token j to source token i
    tgtPills.forEach((tPill, j) => {
      const tRect = tPill.getBoundingClientRect();
      const tx = tRect.left + tRect.width / 2 - wrapperRect.left;
      const ty = height;

      const row = matrix[j] || [];

      srcPills.forEach((sPill, i) => {
        const sRect = sPill.getBoundingClientRect();
        const sx = sRect.left + sRect.width / 2 - wrapperRect.left;
        const sy = 0;

        const weight = row[i] || 0.0;
        if (weight < 0.05) return; // Skip near-zero weights for clean clarity

        // Determine highlighting
        let isHighlighted = false;
        let opacity = Math.min(1.0, Math.max(0.12, weight * 0.9));
        let strokeWidth = Math.max(1.2, weight * 5.0);

        if (activeHoveredSrc !== null) {
          if (activeHoveredSrc === i) {
            isHighlighted = true;
            opacity = Math.max(0.6, weight);
            strokeWidth += 2;
          } else {
            opacity = 0.04;
          }
        } else if (activeHoveredTgt !== null) {
          if (activeHoveredTgt === j) {
            isHighlighted = true;
            opacity = Math.max(0.6, weight);
            strokeWidth += 2;
          } else {
            opacity = 0.04;
          }
        }

        const color = isHighlighted ? '#00f5d4' : (weight > 0.4 ? '#9d4edd' : '#64748b');

        // Smooth cubic bezier S-curve
        const d = `M ${sx} ${sy} C ${sx} ${height * 0.55}, ${tx} ${height * 0.45}, ${tx} ${ty}`;

        paths.push(`
          <path 
            class="ribbon-path" 
            d="${d}" 
            stroke="${color}" 
            stroke-width="${strokeWidth.toFixed(1)}" 
            stroke-opacity="${opacity.toFixed(2)}"
          />
        `);
      });
    });

    el.attentionSvg.innerHTML = paths.join('');
  }

  function redrawRibbons() {
    if (currentData) {
      drawRibbons(
        currentData.src_tokens || [],
        currentData.tgt_tokens || [],
        currentData.attention_matrix || []
      );
    }
  }

  // --- Render 2D Attention Heatmap Grid ---
  function renderHeatmap(srcTokens, tgtTokens, matrix) {
    if (!srcTokens.length || !tgtTokens.length || !matrix.length) {
      el.heatmapGrid.innerHTML = '';
      return;
    }

    let html = '<table class="heatmap-table"><thead><tr><th></th>';
    srcTokens.forEach(s => {
      html += `<th class="heatmap-header src-h">${escapeHtml(s)}</th>`;
    });
    html += '</tr></thead><tbody>';

    tgtTokens.forEach((t, j) => {
      html += `<tr><th class="heatmap-header tgt-h">${escapeHtml(t)}</th>`;
      const row = matrix[j] || [];
      srcTokens.forEach((_, i) => {
        const val = row[i] || 0.0;
        const alpha = Math.min(1.0, val * 0.9 + 0.05);
        const bg = `rgba(157, 78, 221, ${alpha.toFixed(2)})`;
        const textCol = val > 0.4 ? '#ffffff' : 'rgba(255,255,255,0.4)';
        html += `<td class="heatmap-cell" style="background:${bg}; color:${textCol};" title="${val.toFixed(3)}">${val.toFixed(2)}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';

    el.heatmapGrid.innerHTML = html;
  }

  // --- Render Candidate Predictions ---
  function renderCandidates(candidates) {
    if (!candidates.length) {
      el.candidatesList.innerHTML = '<div class="feed-empty">No candidate predictions.</div>';
      return;
    }

    el.candidatesList.innerHTML = candidates.slice(0, 4).map((c, idx) => `
      <div class="cand-item ${c.is_top ? 'top' : ''}">
        <div class="cand-bar" style="width: ${c.confidence_pct}%;"></div>
        <div class="cand-info">
          <span class="cand-rank">#${idx + 1}</span>
          <span class="cand-text">${escapeHtml(c.candidate)}</span>
        </div>
        <span class="cand-pct">${c.confidence_pct}%</span>
      </div>
    `).join('');
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    // Input typing (fast debounced)
    el.textInput.addEventListener('input', (e) => {
      const val = e.target.value.trim();
      debouncedFetch(val);
      updateActiveChip(val);
    });

    // Clear button
    el.clearBtn.addEventListener('click', () => {
      el.textInput.value = '';
      el.textInput.focus();
    });

    // Quick Chips
    el.chipsList.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const word = chip.getAttribute('data-word');
        el.textInput.value = word;
        updateActiveChip(word);
        fetchTelemetry(word);
      });
    });

    // View Toggles: Ribbon vs Heatmap
    el.viewRibbonBtn.addEventListener('click', () => {
      currentView = 'ribbon';
      el.viewRibbonBtn.classList.add('active');
      el.viewMatrixBtn.classList.remove('active');
      el.ribbonStage.style.display = 'flex';
      el.heatmapStage.style.display = 'none';
      redrawRibbons();
    });

    el.viewMatrixBtn.addEventListener('click', () => {
      currentView = 'matrix';
      el.viewMatrixBtn.classList.add('active');
      el.viewRibbonBtn.classList.remove('active');
      el.ribbonStage.style.display = 'none';
      el.heatmapStage.style.display = 'block';
    });

    // Window resize redraws SVG ribbons
    window.addEventListener('resize', () => {
      if (currentView === 'ribbon') redrawRibbons();
    });
  }

  function updateActiveChip(word) {
    el.chipsList.querySelectorAll('.chip').forEach(c => {
      if (c.getAttribute('data-word') === word.toLowerCase()) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // --- Initialization ---
  document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initWebSocket();
  });

})();

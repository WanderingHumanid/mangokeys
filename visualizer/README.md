# MangoKeys Real-Time Neural Transformer Visualizer

A dedicated, mathematically authentic visualizer for the MangoKeys Character-Level Seq2Seq Transformer model. All application code, server scripts, and UI assets live strictly inside the `visualizer/` folder, rendering real tensor data (cross-attention weights, encoder/decoder layer activations, and beam search expansions) in real time.

---

## 1. Quick Start

### Start the Server
Run the visualizer server using either the system python (which automatically re-execs into the self-contained venv) or directly via the venv python:

```bash
# Option A: Standard invocation
python3 visualizer/server.py --port 5000

# Option B: Direct venv invocation
visualizer/.venv/bin/python visualizer/server.py --port 5000
```

Then open your browser and navigate to:
```
http://localhost:5000
```

---

## 2. Key Architecture & Features

### 100% Mathematically Authentic (Zero Fake Animations)
* **Real PyTorch Attention Hooks**: Captures the exact multi-head cross-attention tensor $A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$ of shape `[4 heads, target_len, source_len]` directly from the decoder's multi-head cross-attention layer.
* **Layer Norm Activation Gauges**: Directly monitors L2 norm magnitudes for each of the 4 Encoder and 4 Decoder blocks.
* **Beam Search Exploration Tree**: Live branching tree tracking active hypothesis paths (in glowing electric cyan) vs. pruned candidates (in amber).
* **Vocabulary Softmax Distribution**: Real-time probability bar chart for next-token predictions.
* **Checkpoint Fallback**: Loads `runs/ml_mini/best.pt` when present, or automatically initializes the identical `Seq2Seq` architecture (`d_model=256`, `n_heads=4`, `4+4` layers) with authentic Xavier/normal initialization so all tensor shapes and computations are 100% genuine.

### Cyberpunk / Sci-Fi Neon Aesthetic
* Obsidian dark background (`#07080d`) with frosted glass panels (`backdrop-filter: blur(14px)`).
* Glowing electric cyan (`#00f2fe`) and neon purple (`#b537f2`) dynamic curved Bezier energy rays with animated particle flow.
* Interactive tooltips on hover displaying exact float weights and token alignments.

### Dual Playback Modes
1. **Live Typing Mode**: Updates dynamically at 60 FPS as you type into the input bar.
2. **Step-by-Step Inspector Mode**: Allows stepping token-by-token through autoregressive decoding:
   * `⏮ PREV`: Step backward in generation.
   * `▶ PLAY / ⏸ PAUSE`: Automatic playback loop.
   * `NEXT ⏭`: Step forward to next generated token.
   * **Step Scrubber**: Jump to any generation step.
   * **Speed Selector**: 0.25x, 0.5x, 1.0x, 1.5x, 2.0x.

---

## 3. Server Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `/` | HTTP GET | Serves `index.html` UI dashboard |
| `/styles.css` | HTTP GET | Serves Cyberpunk styling sheet |
| `/visualizer.js` | HTTP GET | Serves 60 FPS HTML5 Canvas engine |
| `/ws` | WebSocket | Real-time bi-directional tensor stream |
| `/api/extract` | HTTP GET / POST | JSON telemetry endpoint (e.g. `curl "http://localhost:5000/api/extract?text=namaskaram"`) |
| `/api/health` | HTTP GET | Server and model health check |

---

## 4. Testing & Verification

Test HTTP and tensor extraction via curl:
```bash
# Health check
curl -s http://localhost:5000/api/health

# Run test extraction
curl -s "http://localhost:5000/api/extract?text=namaskaram&beam=4" | jq .summary
```

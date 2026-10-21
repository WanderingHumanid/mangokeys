# MangoKeys Neural Transformer Visualizer: Architecture & User Guide

A mathematically authentic real-time visualization dashboard for the MangoKeys Character-Level Sequence-to-Sequence Transformer model.

---

## 1. Architectural Philosophy & Mission

### The Core Objective
Provide complete mathematical transparency into how the character-level Transformer transliterates Manglish into Malayalam. Rather than displaying simulated or decorative graphics, every curve, ribbon, heatmap cell, and tree branch in this visualizer is derived directly from authentic ONNX model inference.

### Invariant Design Principles
1. **100% Mathematically Authentic Tensors (Zero Fake Animations)**:
   - Cross-attention ribbons are scaled by the authentic softmax attention matrix:
     $$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$
   - Encoder and decoder layer gauges display real L2 hidden state norm magnitudes.
   - The generation tree tracks genuine autoregressive beam search hypothesis scores and pruned branch trajectories.
   - The candidate distribution displays real top-$k$ output logits over the 133-token Malayalam vocabulary.
2. **Dynamic Power State Management (Idle Standby)**:
   - When no typing activity is occurring, the inference loop transitions into **Idle Standby (0 FLOPs)** to prevent CPU cycles from being wasted on empty cycles.
   - Instantly reactivates upon the first detected keystroke.
3. **Live Desktop Synchronization**:
   - Seamlessly couples with the background `typer` daemon via WebSockets to mirror keystrokes typed across your desktop in real time.

---

## 2. High-Level System Architecture

```
  +-------------------------------------------------------------------------+
  |                          Unified Desktop Source                         |
  |             (Live Desktop Typer Daemon / Browser Input Field)           |
  +-------------------------------------------------------------------------+
                                       |
                                       | (WebSocket / HTTP JSON)
                                       v
  +-------------------------------------------------------------------------+
  |                   Async Server Engine (server.py)                       |
  |  * aiohttp async event loop                                             |
  |  * Real-time WebSocket broadcasting (/ws)                               |
  |  * REST telemetry endpoints (/api/extract, /api/health)                 |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                  Neural Extractor (extractor.py)                        |
  |  * Character-level Tokenizer (TranslitTokenizer)                        |
  |  * ONNX Runtime Inference (encoder.onnx + decoder.onnx)                 |
  |  * Hidden state extraction (d_model = 256, 4 heads, 4+4 layers)         |
  |  * Step-by-step Autoregressive Beam Search Tracer                       |
  +-------------------------------------------------------------------------+
                                       |
                                       | (Telemetry JSON Payload)
                                       v
  +-------------------------------------------------------------------------+
  |                  Frontend Visualization Engine (UI)                     |
  |  * SVG Cross-Attention Ribbons & Multi-Head Isolation                   |
  |  * Step-by-Step Decoder Generation Pipeline Cards                       |
  |  * Interactive Attention Heatmap Matrix                                 |
  |  * Autoregressive Beam Search Exploration Tree                          |
  |  * Dual Mode: Real-time 60 FPS live typing or Step Inspector scrubber   |
  +-------------------------------------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Neural Extraction Engine (`extractor.py`)
- **Model Topology**: Character-level sequence-to-sequence Transformer with 5.37M parameters (`d_model = 256`, `d_ff = 512`, `n_heads = 4`, 4 encoder layers, 4 decoder layers).
- **Execution Mechanism**:
  1. *Encoding*: Processes the source Latin string through `encoder.onnx` to generate memory tensor embeddings.
  2. *Autoregressive Decoding*: Steps through `decoder.onnx` one token at a time, generating output logits for each candidate position.
  3. *Cross-Attention Alignment*: Extracts query-key dot-product distributions linking target Malayalam glyphs to source Latin characters.
  4. *Beam Exploration*: Maintains active vs. pruned hypothesis branches with cumulative log-likelihood scores.

### 3.2 Asynchronous Telemetry Server (`server.py`)
- Built on `aiohttp` for non-blocking bi-directional WebSocket communication.
- **Auto-Venv Re-Execution**: Automatically detects if system Python lacks runtime dependencies (`onnxruntime`, `aiohttp`) and re-executes inside `visualizer/.venv` without requiring manual virtual environment activation.
- **REST & WebSocket Endpoints**:
  - `/ws`: Real-time streaming channel for interactive typing.
  - `/api/extract`: REST endpoint for query extraction and scripting.
  - `/api/health`: Health status, active model path, and memory telemetry.

### 3.3 Interactive UI Engine (`index.html`, `styles.css`, `visualizer.js`)
- **Cyberpunk Dark Aesthetic**: Obsidian surface (`#07080d`), frosted glass panels (`backdrop-filter: blur(14px)`), electric cyan (`#00f2fe`), neon violet (`#b537f2`), and warning amber (`#ff9e00`).
- **Interactive SVG Ribbons**: Smooth cubic Bezier paths visualizing attention flux. Allows isolating individual attention heads (Heads 1 through 4) or viewing the cross-head average.
- **Step Generation Cards**: Sequentially renders each autoregressive prediction step, showing the selected token, its beam score, and the top-4 alternative candidates considered.
- **Heatmap Matrix**: Displays a full 2D token-by-token alignment matrix with interactive tooltips showing exact numerical weights.
- **Dual Playback Modes**:
  - *Live Typing*: Updates instantaneously as keys are pressed.
  - *Step-by-Step Inspector*: Allows pausing, stepping backward/forward (`PREV` / `NEXT`), and scrubbing through the token generation sequence at variable playback speeds (0.25x to 2.0x).

---

## 4. Telemetry Data Contract

The server emits a structured telemetry payload for every transliteration step:

| Property | Type | Description |
| :--- | :--- | :--- |
| `word` | `string` | The input Manglish word (e.g., `"namaskaram"`). |
| `final_text` | `string` | The final generated Malayalam word (e.g., `"നമസ്കാരം"`). |
| `latency_ms` | `float` | Exact inference execution time in milliseconds. |
| `is_idle` | `boolean` | Flag indicating whether the neural engine is in 0-FLOP standby. |
| `frames` | `array` | Array of per-step decoding telemetry records (attention matrices, top candidates, beam hypotheses). |

---

## 5. How to Run

### Method 1: Via the Unified Launcher (Recommended)
Launch via the interactive terminal dashboard:
```bash
python3 launcher.py
```
Select **Option 2 (Visualizer Server)** or run both services together:
```bash
python3 launcher.py --both
```

### Method 2: Direct Execution
```bash
# Launch server on default port 5000
python3 visualizer/server.py --port 5000
```
Open your browser and navigate to:
```
http://localhost:5000
```

### Method 3: Headless API Verification
Query the visualizer engine via HTTP GET / POST:
```bash
# Check server health
curl -s http://localhost:5000/api/health

# Extract telemetry for a sample word
curl -s "http://localhost:5000/api/extract?text=namaskaram&beam=4"
```

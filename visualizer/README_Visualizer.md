# MangoKeys // Real-Time Neural Transformer Visualizer

> Comprehensive technical documentation, architecture specifications, tensor telemetry schemas, and maintenance guides for both human engineers and AI agents working on the MangoKeys Visualizer.

---

## 1. Executive Summary & Philosophy

The **MangoKeys Real-Time Neural Transformer Visualizer** is a high-performance, mathematically authentic visualization dashboard designed specifically for the character-level transliteration Transformer model powering MangoKeys (Manglish to Malayalam).

### Core Design Principles
1. **Zero Fake Animations (100% Authentic Tensors)**:
   Every curve, glow, beam branch, and meter in the visualizer is driven by authentic PyTorch tensor computations during the forward pass.
   - Cross-attention curves are scaled directly by the softmax attention matrix $A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$ of shape `[num_heads=4, tgt_len, src_len]`.
   - Encoder and decoder block pulses are driven by actual layer L2 activation norms.
   - The tree displays genuine beam search log probabilities and real pruned candidate trajectories.
   - The bar chart displays actual top-$k$ softmax output logits over the 133-character Malayalam vocabulary.
2. **Self-Contained Subsystem**:
   All server code, tensor extraction logic, client JavaScript, styles, markup, and dedicated virtual environment live strictly inside the `visualizer/` folder.
3. **Cyberpunk / Sci-Fi Neon Aesthetic**:
   Obsidian black background (`#07080d`), frosted glass panels with backdrop blur (`backdrop-filter: blur(14px)`), electric cyan (`#00f2fe`), neon purple (`#b537f2`), and warning amber (`#ff9e00`).
4. **Dual Playback Ergonomics**:
   - **Live Typing Mode**: Updates dynamically at 60 FPS as the user types into the input bar.
   - **Step-by-Step Inspector Mode**: Allows pausing autoregressive decoding to step forward, backward, scrub through time, and adjust playback speeds (0.25x – 2.0x).

---

## 2. Directory Structure

```
visualizer/
├── .venv/                   # Dedicated virtualenv (PyTorch CPU, aiohttp, numpy)
├── extractor.py             # PyTorch hook engine & beam search tracer
├── server.py                # Async HTTP & WebSocket streaming server
├── index.html               # Main cyberpunk dashboard UI
├── styles.css               # Sci-Fi glassmorphic styling system
├── visualizer.js            # 60 FPS HTML5 Canvas rendering engine
├── README.md                # Quickstart instructions
└── README_Visualizer.md     # This comprehensive developer reference
```

---

## 3. Deep Dive: Tensor Extractor Engine (`extractor.py`)

The extractor wraps `model.src.seq2seq.Seq2Seq` and `model.src.tokenizer.TranslitTokenizer`.

### 3.1 Architecture Specifications
- **Model**: Character-Level Sequence-to-Sequence Transformer
- **Parameters**: 5,372,293 (~5.37M)
- **Embedding Dimension ($d_{\text{model}}$)**: 256
- **Feed-Forward Dimension ($d_{\text{ff}}$)**: 512
- **Attention Heads ($n_{\text{heads}}$)**: 4 (Head Dimension $d_k = 64$)
- **Layers**: 4 Encoder Layers + 4 Decoder Layers
- **Source Vocabulary**: 99 tokens (ASCII printable + specials)
- **Target Vocabulary**: 133 tokens (Malayalam block + specials)

### 3.2 PyTorch Hook Architecture
To capture genuine intermediate tensors without altering the original model code in `model/src/`, `extractor.py` registers forward hooks:

```python
# 1. Cross-Attention Hook (Decoder Layer 3)
def make_cross_attn_hook(layer_i: int):
    def hook(module, inputs, output):
        # inputs: (query, key, value, attn_mask)
        query, key = inputs[0], inputs[1]
        attn_mask = inputs[3] if len(inputs) > 3 else None

        q = module._split(module.q_proj(query))
        k = module._split(module.k_proj(key))
        scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        self.captured_cross_attn[layer_i] = weights.detach().cpu()
    return hook

dec_layer.cross_attn.register_forward_hook(make_cross_attn_hook(idx))
```

- **Encoder & Decoder Activations**: Forward hooks calculate the mean L2 norm of the hidden representation across tokens:
  $$\text{Norm} = \frac{1}{T} \sum_{t=1}^T \| h_t \|_2$$
  These values are normalized into $[0.25, 0.98]$ for UI gauge meters.

### 3.3 Checkpoint Fallback Mechanism
When initialized:
1. It attempts to load pre-trained weights from `runs/ml_mini/best.pt`.
2. If `runs/ml_mini/best.pt` does not exist, it instantiates an initialized `Seq2Seq` instance with the identical configuration (`d_model=256`, `n_heads=4`, `d_ff=512`, `4+4` layers).
3. Because the forward pass, projection matrices, softmax equations, and shapes are identical, all attention weights and beam probabilities are 100% mathematically authentic and never mocked.

### 3.4 Step-by-Step Beam Search Tracer
`TensorExtractor.trace(text: str, beam_width: int = 4, max_steps: int = 24)` executes autoregressive decoding while recording per-step telemetry frames:
- **Encoding**: Generates memory tensor `memory = model.encode(src, src_mask)`.
- **Active Beams Pool**: Tracks candidate hypotheses with cumulative log probabilities:
  $$\text{Score}_{\text{new}} = \text{Score}_{\text{parent}} + \log P(y_t \mid y_{<t}, x)$$
- **Surviving vs. Pruned Nodes**: The top $k$ candidates survive into the active beam pool (`pruned: false`), while candidates ranked immediately below the beam threshold are recorded as pruned branches (`pruned: true`).
- **Telemetry Frame Generation**: Produces a frame record at every decoding step $t$.

---

## 4. Deep Dive: Async Server (`server.py`)

`visualizer/server.py` is built using Python's `aiohttp` framework.

### 4.1 Auto-Re-Execution Mechanism
If an operator runs `python3 visualizer/server.py` using a system Python that lacks PyTorch, the script detects the missing module and seamlessly re-execs itself using the dedicated virtual environment:
```python
try:
    import aiohttp
    import torch
except ImportError:
    venv_py = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if venv_py.exists() and sys.executable != str(venv_py):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    raise
```

### 4.2 Endpoint Specifications

| Route | Method / Protocol | Description | Payload / Response |
|---|---|---|---|
| `/` | `HTTP GET` | Serves dashboard HTML | `index.html` (`text/html`) |
| `/styles.css` | `HTTP GET` | Serves stylesheet | `styles.css` (`text/css`) |
| `/visualizer.js` | `HTTP GET` | Serves client engine | `visualizer.js` (`application/javascript`) |
| `/api/health` | `HTTP GET` | Health check & model status | JSON status, parameter count, layer configuration |
| `/api/extract` | `HTTP GET / POST` | REST extraction endpoint | Query params or JSON body: `text`, `beam`, `max_steps` |
| `/ws` | `WebSocket` | Real-time bi-directional stream | Bi-directional JSON message protocol |

### 4.3 WebSocket Protocol
- **Client to Server**:
  ```json
  {
    "type": "input",
    "text": "namaskaram",
    "beam": 4,
    "max_steps": 24
  }
  ```
- **Server to Client**:
  ```json
  {
    "type": "telemetry",
    "word": "namaskaram",
    "final_text": "നമസ്കാരം",
    "latency_ms": 32.4,
    "frames": [ /* telemetry frames */ ],
    "summary": { "num_steps": 24, "encoder_layers": 4, "decoder_layers": 4, "heads": 4 }
  }
  ```

---

## 5. Telemetry Frame JSON Schema

Each frame in `frames[]` corresponds to one autoregressive decoding step:

```json
{
  "active_step": 3,
  "input_tokens": ["n", "a", "m", "a", "s", "k", "a", "r", "a", "m"],
  "output_tokens": ["ന", "മ", "സ്"],
  "cross_attention": [
    // 4 heads x target_len (3) x source_len (10)
    [
      [0.8241, 0.0412, 0.0211, ...],
      [0.0123, 0.7412, 0.1021, ...],
      [0.0051, 0.0112, 0.8924, ...]
    ],
    // Head 1, Head 2, Head 3...
  ],
  "layer_activations": {
    "encoder": [0.624, 0.766, 0.914, 0.980],
    "decoder": [0.630, 0.766, 0.863, 0.980]
  },
  "beam_tree": [
    {"id": "root", "parent": null, "token": "<bos>", "score": 0.0, "pruned": false, "step": 0},
    {"id": "b_1_0", "parent": "root", "token": "ന", "score": -0.124, "pruned": false, "step": 1},
    {"id": "p_1_0", "parent": "root", "token": "ണ", "score": -1.842, "pruned": true, "step": 1}
  ],
  "top_k_candidates": [
    {"char": "്", "prob": 0.7241},
    {"char": "ാ", "prob": 0.1824},
    {"char": "ി", "prob": 0.0512}
  ]
}
```

---

## 6. Frontend Engine & Canvas Rendering (`visualizer.js`)

### 6.1 Canvas Architecture & Retina Support
Two primary HTML5 Canvas contexts run at 60 FPS via `requestAnimationFrame`:
1. `attentionCanvas`: Visualizes Latin source tokens, middle Transformer pipeline blocks, Malayalam target tokens, and dynamic attention Bezier curves with particle trails.
2. `beamTreeCanvas`: Visualizes the branching beam search exploration tree.

Both canvases automatically handle display scaling using `devicePixelRatio` to eliminate blurriness on Retina and 4K displays:
```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = clientWidth * dpr;
canvas.height = clientHeight * dpr;
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
```

### 6.2 Attention Curves & Head Filtering
- **Head Filtering**:
  - `state.selectedHead = -1`: Averages attention weights across all 4 heads:
    $$\bar{A}_{ij} = \frac{1}{4} \sum_{h=0}^3 A_{h, i, j}$$
  - `state.selectedHead = 0..3`: Isolates Head 1, 2, 3, or 4 to inspect head specialization (e.g. alignment heads vs. phonetic contextual heads).
- **Cubic Bezier Path**:
  Curves connect source token $(x_s, y_s + 16)$ to target token $(x_t, y_t - 16)$ with control points aligned vertically to the pipeline layer:
  - $P_0 = (x_s, y_s + 16)$
  - $P_1 = (x_s, y_{\text{mid}} - 20)$
  - $P_2 = (x_t, y_{\text{mid}} + 20)$
  - $P_3 = (x_t, y_t - 16)$
- **Visual Mapping**:
  - Ray Line Width: $W = \max(0.8, \text{weight} \times 4.5)$
  - Opacity: $\alpha = \min(1.0, 0.15 + \text{weight} \times 0.85)$
  - Glow Blur: Intense cyan/purple glow when $\text{weight} > 0.30$.

### 6.3 Dynamic Photon Particle Trails
For rays with attention weights $> 0.15$, particles spawn and travel along the Bezier curve:
$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3, \quad t \in [0, 1]$$
Particle velocity and luminosity are directly proportional to the attention weight.

### 6.4 Hierarchical Beam Tree Layout
`renderBeamTree()` assigns coordinates by step:
- $X = \text{padLeft} + \text{step} \times \Delta X$
- $Y = \text{padTop} + \frac{\text{index} + 1}{N + 1} \times \Delta Y$
Curved connectors link child nodes to parent hypotheses:
- Active surviving beams: solid electric cyan with cumulative score tags.
- Pruned hypotheses: dashed amber trails (`#ff9e00`).

---

## 7. Developer & Agent Guide: How to Extend or Update

### 7.1 Testing the Server Locally
To launch the server on port 5000:
```bash
python3 visualizer/server.py --port 5000
```

To run a headless test verifying endpoints and tensor calculation:
```bash
# Health check
curl -s http://127.0.0.1:5000/api/health

# Test tensor extraction
curl -s "http://127.0.0.1:5000/api/extract?text=namaskaram&beam=4" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data['type'] == 'telemetry'
assert len(data['frames']) > 0
print('Verification successful! Frames generated:', len(data['frames']))
"
```

### 7.2 Connecting a Newly Trained Model Checkpoint
To point the visualizer to a new checkpoint (e.g. `runs/e1_standard_only/best.pt` or `runs/ml_prefix/best.pt`):
1. In [`visualizer/server.py`](file:///home/vishnunandan555/Projects/mangokeys/visualizer/server.py), update the `checkpoint_path`:
   ```python
   self.extractor = TensorExtractor(
       checkpoint_path=WORKSPACE_ROOT / "runs/e1_standard_only/best.pt",
       tokenizer_path=WORKSPACE_ROOT / "runs/e1_standard_only/tokenizer.json",
   )
   ```
2. The extractor will automatically read `state_dict` and update model weights.

### 7.3 Adding New Attention Heads or Layers
If updating the model architecture (e.g. $n_{\text{heads}} = 8$ or 6 layers):
1. In [`visualizer/extractor.py`](file:///home/vishnunandan555/Projects/mangokeys/visualizer/extractor.py): Update `n_heads`, `n_enc_layers`, and `n_dec_layers` in `_init_model()`.
2. In [`visualizer/index.html`](file:///home/vishnunandan555/Projects/mangokeys/visualizer/index.html): Update the `#headSelector` buttons with the new head count.
3. In [`visualizer/visualizer.js`](file:///home/vishnunandan555/Projects/mangokeys/visualizer/visualizer.js): The renderer dynamically adapts to whatever `cross_attention.length` heads are returned in the telemetry JSON.

### 7.4 Strict Boundary Rules
- **Rule**: Work EXCLUSIVELY inside the `visualizer/` folder.
- **Rule**: Do NOT modify files outside `visualizer/` (such as `model/src/`, `data/`, or `typer/`).

# mangokeyboard

**Manglish → മലയാളം transliteration.** A college ML project: a compact character-level
seq2seq model, a study of *how Malayalam users actually type*, and a working
on-device demo.

**[→ Full implementation plan: `IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)**

---

## The project in one line

Real Manglish typing is not ISO 15919. Users drop inherent vowels, collapse gemination, and
substitute chillus. A model trained on clean pairs benchmarks beautifully and disappoints users.
**The contribution is quantifying that gap and fixing it.**

| | |
|---|---|
| **The gap** | Same model, same data — **7.4% CER** on careful typing → **22.7% CER** on how people really type |
| **The method** | A 13-rule augmentation pipeline that manufactures the missing distribution |
| **The model** | 5.37M-param char-level Transformer enc-dec → **~5.4 MB** int8 ONNX |
| **The baselines** | [IndicXlit](https://github.com/AI4Bharat/IndicXlit) (MIT, 11M) · our own rule engine · the published Bi-LSTM (IndoNLP-2025) |
| **The demo** | A web page (in-browser, offline) + a sideloaded Android APK with zero permissions |
| **Cost** | **₹0–2,000** · 8 weeks · a single Colab T4 is enough to train |

> **Rule baseline, measured on the Aksharantar Malayalam test set (12,451 words):**
> Top-1 **9.2%**, CER **48.8%**. The rules cannot resolve the ambiguities that Malayalam
> orthography hides (ന്ദ vs ണ്ട, ശ vs ഷ vs സ, ന്ത vs ന്റ). That gap is what the model is for.

## Status

| Piece | State |
|---|---|
| Data pipeline (download → normalize → filter → augment → split) | ✅ working on the real 4.1M-pair corpus |
| Model, training loop, length bucketing, label smoothing | ✅ working |
| Augmentation rules (13) with unit tests | ✅ working |
| Evaluation (CER / Top-1 / Top-5, grapheme aware, per bucket) | ✅ working |
| Rule-based baseline + baseline evaluation | ✅ working |
| ONNX export, int8 quantization path, Python runtime | ✅ working, parity checked against PyTorch |
| Browser runtime (ONNX Runtime Web) | ✅ written |
| Kotlin runtime (ONNX Runtime Mobile) | ✅ written |
| Model trained on the full corpus | ⏳ running |

## Layout

```
model/       training, evaluation, ONNX export, quantization
  src/       tokenizer · normalization · seq2seq · data · decode · metrics · rules · augment
data/        download · normalize · filter · augment · split scripts
runtime/     python · kotlin (Android) · js (browser) inference
scripts/     parity check · baselines · ablations · demo
evals/       adhoc manual set · mozhi conformance · results
tests/       unit tests for every module
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Fetch the data (about 130 MB for the Malayalam slice)
python -m data.scripts.download_aksharantar --out data/raw

# 2. Build the splits, with augmentation
python -m data.scripts.build_splits --out data/processed

# 3. Train
python -m model.train --config configs/ml_mini.yaml

# 4. Evaluate against the rule baseline
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data data/processed/test_aksharantar.tsv
python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv

# 5. Export and check that the ONNX runtime matches PyTorch
python -m model.export_onnx --checkpoint runs/ml_mini/best.pt --out runs/ml_mini/onnx
python -m scripts.check_parity --checkpoint runs/ml_mini/best.pt --onnx runs/ml_mini/onnx

# 6. Try it
python -m scripts.demo --onnx runs/ml_mini/onnx --tokenizer runs/ml_mini/tokenizer.json --word ente
```

## Credits

Full attribution table in [`LICENSES.md`](LICENSES.md).

- **Aksharantar** (AI4Bharat) — CC-BY / CC0 — `huggingface.co/datasets/ai4bharat/Aksharantar`
- **Dakshina** (Google) — CC BY-SA 4.0 — `github.com/google-research-datasets/dakshina`
- **IndicXlit** (AI4Bharat) — MIT — `github.com/AI4Bharat/IndicXlit`
- **Mozhi 2.0** romanization spec — open spec
- **IndoNLP-2025** shared task test sets — evaluation only

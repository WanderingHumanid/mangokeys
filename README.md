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
| Model, training loop, length bucketing, label smoothing, AMP | ✅ 5.37M params, trained on 6.6M pairs |
| Augmentation rules (15) with unit tests | ✅ working |
| Prefix training pairs (the model can stop in the middle of a word) | ✅ working |
| Evaluation (CER / Top-1 / Top-5, grapheme aware, per bucket) | ✅ working |
| Rule-based baseline | ✅ Top-1 9.2%, passes all 67 Mozhi cases |
| Word list with a deletion index | ✅ 118K words, the model with it is +4 points |
| Character language model | ✅ trained, measured, and it did not help (off by default) |
| ONNX export, int8 quantization, Python runtime | ✅ parity checked against PyTorch |
| Browser runtime (ONNX Runtime Web) | ✅ verified in a real browser, agrees with Python |
| Kotlin runtime (ONNX Runtime Mobile) | ✅ builds and passes the unit tests against the real model |
| Static web demo with a suggestion strip | ✅ working, screenshot verified |
| Android input method (an `InputMethodService`) | ✅ builds, installs, and asks for no permissions |
| Error analysis | ✅ 12-class taxonomy, dumped to `evals/results/errors.tsv` |
| Augmentation ablation | ✅ measured |

### Measured on the Aksharantar Malayalam test set

| System | Top-1 | CER | Top-5 | Size | ms / word |
|---|---|---|---|---|---|
| Rule engine (ours) | 9.2% | 48.8% | 13.6% | 3 KB | 0.10 |
| **Our model, int8, beam 4** | **53.4%** | **17.0%** | **64.6%** | **6.1 MB** | 27.4 |
| **Our model with the word list** | **57.4%** | **15.9%** | - | 6.1 MB | 33 |
| Our model, int8, greedy | 53.0% | 17.1% | - | 6.1 MB | **12.0** |

The model is 5.8 times better than the rule engine on top-1, and its character
error rate is 2.9 times lower. Quantization to int8 costs nothing measurable.
The timings are the int8 model on a desktop cpu, which is what the browser and
the phone use.

**The word list is the keyboard.** A model that transliterates a word is not a
keyboard. Manglish Keyboard is a ranking system: the user types a rough
spelling and the keyboard offers a list of the words it could mean. The word
list finds the real words around the model's answer, and the model's answer
is right 87% of the time when the list knows the word and 51% of the time when
it does not, so the list is also a confidence signal. The report has the
measurements, including the ones that went against us.

In the browser the model runs in a Web Worker on four threads, so the page
never stutters: the longest frame gap while typing is 17 to 27 milliseconds,
which is the normal frame budget. See [`docs/LATENCY.md`](docs/LATENCY.md).

### The augmentation is the whole project

Two models, same architecture, same row count, same epochs, same seed. Only
the style of the extra rows differs.

| Training data | Standard test set | | Adhoc test set | |
|---|---|---|---|---|
| | Top-1 | CER | Top-1 | CER |
| standard rows only | **56.2%** | **15.4%** | 6.2% | 49.7% |
| standard + adhoc rows | 53.4% | 17.0% | **46.2%** | **20.0%** |

Training on careful spelling alone makes a model three points better at
careful spelling and **40 points worse at real typing**. That is the
difference between a keyboard that works and one that does not.

The model has **solved the letter ambiguities** (ശ/ഷ/സ, ട/ത, ര/റ, ല/ള, ന/ണ and
the chillus together are 5.7% of its errors) and has **not solved the
lengths** (missing or extra vowel signs and viramas are 67%). The error
analysis in `scripts/error_analysis.py` is what found that, and it is what
motivated the long vowel augmentation rules.

## Status

| Piece | State |
|---|---|
| Data pipeline (download → normalize → filter → augment → split) | ✅ working on the real 4.1M-pair corpus |
| Model, training loop, length bucketing, label smoothing, AMP | ✅ 5.37M params, trained on 6.6M pairs |
| Augmentation rules (15) with unit tests | ✅ working |
| Prefix training pairs, so the model can stop in the middle of a word | ✅ working |
| Evaluation (CER / Top-1 / Top-5, grapheme aware, per bucket) | ✅ working |
| Rule-based baseline | ✅ Top-1 9.2%, passes all 67 Mozhi cases |
| Word list with a deletion index | ✅ 118K words, the model with it is four points better |
| Character language model | ✅ trained and measured, and it did not help (off by default) |
| ONNX export, int8 quantization, Python runtime | ✅ parity checked against PyTorch |
| Browser runtime (ONNX Runtime Web) | ✅ verified in a real browser, agrees with Python |
| Kotlin runtime (ONNX Runtime Mobile) | ✅ builds and passes the unit tests against the real model |
| Static web demo with a suggestion strip | ✅ working, screenshot verified |
| Android input method, an `InputMethodService` | ✅ builds, installs, and asks for no permissions |
| Error analysis | ✅ 12-class taxonomy, dumped to `evals/results/errors.tsv` |
| Augmentation ablation | ✅ measured |

## Layout

```
model/       training, evaluation, ONNX export, quantization
  src/       tokenizer · normalization · seq2seq · data · decode · metrics · rules · augment
             lexicon · ngram · prefix · suggest
data/        download · normalize · filter · augment · split scripts
runtime/     python · kotlin (Android) · js (browser) inference
apps/        expo-web (the demo) · android (the input method)
scripts/     parity check · baselines · ablations · demo · tuning
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

# 3. Build the word list and the language model
python -m data.scripts.download_frequency --out data/raw/frequency
python -m data.scripts.download_dakshina_lexicon --out data/raw/dakshina
python -m scripts.build_lexicon --out data/processed/lexicon
python -m scripts.train_ngram --max-words 400000 --out data/processed/ngram.txt

# 4. Train
python -m model.train --config configs/ml_mini.yaml

# 5. Evaluate against the rule baseline
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data data/processed/test_aksharantar.tsv
python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv

# 6. Export and check that the ONNX runtime matches PyTorch
python -m model.export_onnx --checkpoint runs/ml_mini/best.pt --out runs/ml_mini/onnx
python -m scripts.check_parity --checkpoint runs/ml_mini/best.pt --onnx runs/ml_mini/onnx

# 7. Try it
python -m scripts.demo --onnx runs/demo/onnx-int8 --tokenizer runs/ml_mini/tokenizer.json --word ente

# 8. Build and serve the expo demo
python -m scripts.build_demo --checkpoint runs/ml_mini/best.pt --quantize
python apps/expo-web/serve.py --port 8000
# then open http://localhost:8000

# 9. Build and test the android runtime and the input method
cd runtime && export ANDROID_HOME=~/Android/Sdk && ./gradlew test
./gradlew :app:assembleDebug
adb install -r ../apps/android/build/outputs/apk/debug/app-debug.apk
```

[`TESTING.md`](TESTING.md) has the full procedure for every module, on the
emulator and on a phone.

## Credits

Full attribution table in [`LICENSES.md`](LICENSES.md).

- **Aksharantar** (AI4Bharat) — CC-BY / CC0 — `huggingface.co/datasets/ai4bharat/Aksharantar`
- **Dakshina** (Google) — CC BY-SA 4.0 — `github.com/google-research-datasets/dakshina`
- **IndicXlit** (AI4Bharat) — MIT — `github.com/AI4Bharat/IndicXlit`
- **Mozhi 2.0** romanization spec — open spec
- **IndoNLP-2025** shared task test sets — evaluation only

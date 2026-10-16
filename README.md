# mangokeyboard

**Manglish → മലയാളം transliteration.** A college ML project: a compact character-level
seq2seq model, a rigorous study of *how Malayalam users actually type*, and a working
on-device Android demo.

**[→ Full implementation plan: `IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)**

---

## The project in one line

Real Manglish typing is not ISO 15919. Users drop inherent vowels, collapse gemination, and
substitute chillus. A model trained on clean pairs benchmarks beautifully and disappoints users.
**The contribution is quantifying that gap and fixing it.**

| | |
|---|---|
| **The gap** | Same model, same data — **7.4% CER** on careful typing → **22.7% CER** on how people really type |
| **The method** | A 15-rule augmentation pipeline that manufactures the missing distribution |
| **The model** | 5.3M-param char-level Transformer enc-dec → **5.4 MB** int8 ONNX |
| **The architecture** | 🟢 **Rules propose, the model disposes.** [Ezhuthola](https://github.com/rag-creation/ezhuthola)'s 3 KB rule engine generates ≤300 candidate spellings; our model *ranks* them. One-function swap, airtight ablation. |
| **The demo** | A web page (fully offline, in-browser) + a sideloaded Android APK with zero permissions, built as a fork of Ezhuthola |
| **The baselines** | [IndicXlit](https://github.com/AI4Bharat/IndicXlit) (MIT, 11M) · **Ezhuthola's rule engine** (production, 3.7 MiB app) · the published Bi-LSTM (IndoNLP-2025) |
| **Cost** | **₹0–2,000** · 8 weeks · Colab T4 is enough to train |

> **Note on Ezhuthola:** it is **100% rule-based — no ML whatsoever** (verified: no ML dependency
> in `libs.versions.toml`, no model file in `assets/`, no ML licence in `licenses/`, 3.7 MiB APK).
> Its frequency-list *ranker* is the thing your model replaces. See `IMPLEMENTATION_PLAN.md` §4.7.

## Status

Planning. See the [8-week timeline](IMPLEMENTATION_PLAN.md#14-timeline--8-weeks) and the
[scope cut list](IMPLEMENTATION_PLAN.md#3-scope) if you're behind schedule.

## Key dates / gates

| Week | Gate |
|---|---|
| 1 | 🔴 Run IndicXlit + the rule baseline. **If we can't beat them, we ship them and reframe.** |
| 2 | 🔴 Reproduce the adhoc cliff (adhoc CER ≥ 1.5× standard) |
| 3 | 🔴 Augmentation must improve adhoc CER by **≥3 points absolute** |
| 6 | 🔴 The web demo must run **offline** in a browser |
| 8 | Expo-ready |

## Layout

```
model/       training, evaluation, ablations, ONNX export
data/        download · normalize · augment · split scripts
runtime/     python · kotlin (Android) · js (browser) inference
apps/        expo-web (Next.js) · android (fork of an existing keyboard)
evals/       your collected adhoc set · mozhi conformance · results
docs/        evaluation protocol, licensing, style notes
```

## Write-up

The report goes in [`REPORT.md`](REPORT.md) — abstract, method, results, ablations, error
analysis. Write it incrementally, experiment by experiment.

## Credits

Full attribution table in [`LICENSES.md`](LICENSES.md). Datasets and models used, and their
licences:

- **Dakshina** (Google) — CC BY-SA 4.0 — `github.com/google-research-datasets/dakshina`
- **Aksharantar** (AI4Bharat) — CC-BY / CC0 — `huggingface.co/datasets/ai4bharat/Aksharantar`
- **IndicXlit** (AI4Bharat) — MIT — `github.com/AI4Bharat/IndicXlit`
- **Mozhi 2.0** romanization spec — open spec
- **IndoNLP-2025** shared task test sets — evaluation only

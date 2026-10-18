# Evaluation Protocol

The single source of truth for how this project measures itself. Every model card, every
PR description, and every `eval.json` in `evals/results/` is produced by running this protocol
end to end. **No metric is reported unless it comes out of this pipeline.**

---

## 1. Metric definitions

| Metric | Definition | Notes |
|---|---|---|
| **CER** | `Σ Levenshtein(hyp, ref) / Σ len(ref)` — micro-averaged over the set | Primary metric. Standard in this literature (Dakshina, Bajiyo et al.). |
| **WER** | Word-level error rate | Secondary; matters for the translator use case. |
| **Top-1** | `fraction(hyp == ref)` — exact string match | **The headline number.** This is what Dakshina/IndicXlit report. |
| **Top-5 recall** | `fraction(ref ∈ top5_candidates)` | What matters for a *suggestion strip* — the user picks from a list. |
| **Round-trip** | `roman → native → roman == roman` | Cheap correctness proxy; catches malformed output. |
| **p50 / p95 latency** | ms per word, measured on target hardware | Product metric. |
| **Model size** | MB on disk | Product metric. |

**Levenshtein** is computed on **grapheme clusters**, not raw codepoints. Malayalam's vowel signs
and virama are combining marks — counting them as separate characters makes CER meaningless and
makes two different models look identical. Use a grapheme-aware segmentation (e.g. `regex`
with the `\X` flag, or `grapheme` package) for both hypothesis and reference.

---

## 2. Evaluation sets

| Set id | Source | n | Purpose |
|---|---|---|---|
| `test-standard` | Aksharantar `mal` test | 12,500 | Held-out general transliteration |
| `test-adhoc` | **IndoNLP-2025 Test Set-2** | — | 🔴 **North star.** Adhoc typing style |
| `test-standard-ext` | IndoNLP-2025 Test Set-1 | — | 🟠 North-star control |
| `test-dakshina` | Dakshina ml dev + test lexicon | 5,000 | 🟠 Head-to-head with published numbers |
| `test-ne` | Aksharantar AK-NEF + AK-NEI (ml) | 835 + 1,219 | 🟡 Named entities (hard) |
| `test-mozhi` | Mozhi 2.0 spec test cases | hundreds | 🟡 Scheme conformance |
| `test-manual` | Native-speaker adhoc set | 500 → 2,000 | 🟡 Ground truth for style |

**`test-adhoc` is the north star.** A model can look excellent on every other set and still be
mediocre in a real keyboard. The published Bi-LSTM baseline is **7.4% CER standard / 22.7% adhoc** —
the gap between those two numbers *is* the problem we are solving.

---

## 3. Required report format

Every evaluation run emits a row per set:

```
set                  n       Top-1   Top-5    CER      WER     RT-acc
test-standard        12500   0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-adhoc           N       0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-dakshina        5000    0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-ne              2054    0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-manual          N       0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
```

**Mandatory companion rows:**

- **IndicXlit baseline** — the same table, from the same harness. If we cannot beat it, we ship it.
- **Rule-based baseline** — the deterministic layer on its own. This tells us how much the neural
  model actually contributes.
- **Ours, int8** — the quantized artifact, not just the fp32 checkpoint. Quantization can cost
  accuracy; the shipped number is the one that matters.

### Leakage check

The Aksharantar train and test splits share **zero** native words. Checked
directly over the 4.1M pair training pool and the 12,451 row test set, in
every subset:

| Test subset | native words also in train |
|---|---|
| AK-Freq | 0 / 4858 |
| AK-NEF | 0 / 833 |
| AK-NEI | 0 / 1191 |
| Dakshina | 0 / 5569 |

The augmented copies do not change this, because augmentation rewrites the
roman side and never introduces a new native word.

---

## 4. Bucketed breakdown

Aggregate CER hides the failures that matter. Every run also reports:

| Bucket | Definition | Why |
|---|---|---|
| **Input length** | roman length 1–3 / 4–6 / 7–9 / 10+ | Long words accumulate more error |
| **Frequency** | head / mid / tail by training-corpus frequency | Tests memorisation vs. generalisation |
| **In-vocab** | roman form appears in the training corpus | The dictionary tier handles these |
| **OOV** | roman form unseen | The model tier is the only hope — this is where quality shows |
| **Contains chillu** | reference contains ൺ ൻ ർ ൽ ൾ ൿ ൔ ൕ ൖ | Unicode-normalisation failures surface here |
| **Contains geminate** | doubled consonant in reference | Sandhi / gemination errors |
| **Contains cluster** | conjunct (്) in reference | Conjunct formation errors |
| **Contains retroflex** | ട ഠ ഡ ഢ ണ | The `T`/`t` case distinction |
| **Postposition suffix** | token ends in a known Malayalam postposition | Tests the morphology module (§7.4) |
| **Latin loanword** | reference contains a loanword | `four` → ഫോർ class |

**The OOV bucket is the one to watch.** In the shipped product, in-vocab words hit the dictionary
tier and never reach the model. So the model's real-world quality is approximately its **OOV
performance**, not its aggregate score. Report OOV separately, always.

---

## 5. Error analysis protocol

For every run, dump the top-50 worst errors to `evals/results/errors/<run_id>.tsv` with columns:
`roman`, `reference`, `hypothesis`, `cer`, `bucket`, `in_vocab`, `freq`.

Manually classify at least the top 100 into a fixed taxonomy:

| Class | Example |
|---|---|
| `vowel-drop` | `sundari` → സുന്ദറി (expected സുന്ദരി) |
| `gemination` | `kat` → കത്ത് (expected കത്ത്) |
| `chillu` | `ml` → മ്ല (expected മൽ) |
| `cluster` | `ksh` → ക്ഷ (expected ക്ശ) |
| `retroflex` | `tha` → ഥ (expected ത) |
| `sh-vs-s` | `s` → സ (expected ശ) |
| `zh-vs-l` | `zh` → ഴ (expected ള) |
| `morphology` | `keralathil` → കേരളതിൽ (expected കേരളത്തിൽ) |
| `segmentation` | compound split error |
| `ne-unknown` | named entity not in training |
| `latin-leak` | output contains Latin characters |

**If a class accounts for >20% of errors, it gets a targeted data fix in the next iteration.**
This closes the loop between evaluation and the augmentation pipeline.

---

## 6. Regression gate (CI)

`scripts/run_all_evals.sh` runs on every push that touches `model/`, `data/`, or `evals/`.

**Blocking gates** (CI fails, PR blocked):
- `test-standard` CER must not regress > **0.5 points** absolute vs. `main`.
- `test-adhoc` CER must not regress > **1.0 point** absolute vs. `main`.
- Top-1 on `test-dakshina` must not regress vs. `main`.
- ONNX fp32 parity: max absolute logit difference vs. PyTorch < **1e-4**.
- int8 accuracy delta on `test-adhoc` vs. fp32 must be < **1.0 point** CER.

**Non-blocking (warn only):**
- `test-ne` regression > 3 points → flag in the PR comment; NE is noisy and small.

---

## 7. Reproducibility

- Every eval run records: git SHA, config file, checkpoint SHA256, dataset version/hash,
  tokenizer version, seed, runtime (PyTorch/ORT) versions.
- `evals/results/eval.json` is committed and regenerated by CI; the expo web demo reads it
  directly, so the published numbers are always the committed numbers.
- Seeds: 3 seeds for the final model; report mean ± std. Single-seed numbers are marked as such.

---

## 8. Model card minimum contents

- Training data table with licence per source and the exact row counts used
- Eval tables (all three tiers: fp32 / int8 / rule baseline)
- Bucketed breakdown
- Latency on the target device (model, not framework)
- Model size, and the exact file the user downloads
- Known failure modes, from the error taxonomy above
- Licence of the weights, and of every data source

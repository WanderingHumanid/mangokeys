# Implementation Plan — Malayalam / Manglish Transliteration Mini-Model

**Project type:** College ML project (demo + expo exhibition)
**Project:** `mangokeyboard`
**Status:** Planning (greenfield)
**Date:** 2026-10-20

**Original brief (section 3 of the source doc):**
> Build a small character-level seq2seq model on Google's Dakshina dataset (which includes Malayalam romanization pairs). It's only worth it if you're already doing the Manglish translator — in that case, ship a tiny version for the expo instead of starting something new.

**What this plan delivers:** a trained char-level transliteration model, a rigorous evaluation of the *adhoc* typing gap, and a working on-device Android demo — in ~8 weeks, for ~₹0–2,000.

---

## 0. Status

The project is built and running. This section records where each part stands.

| Component | State | Where |
|---|---|---|
| Data pipeline | ✅ working on the real 4.1M-pair corpus | `data/scripts/` |
| Augmentation rules (15) | ✅ with unit tests | `model/src/augment.py` |
| Model, training, AMP, length bucketing | ✅ 5.37M params, 6.6M pairs, 12 epochs | `model/src/seq2seq.py` |
| Evaluation, grapheme aware CER | ✅ | `model/src/metrics.py` |
| Rule-based baseline | ✅ Top-1 9.2%, CER 48.8%, 67/67 Mozhi cases | `model/src/rules.py` |
| ONNX export, int8 path, Python runtime | ✅ parity checked against PyTorch | `model/export_onnx.py` |
| Browser runtime | ✅ verified in a real browser, agrees with Python | `runtime/js/` |
| Kotlin runtime | ✅ written | `runtime/kotlin/` |
| Static web demo | ✅ working, screenshot verified | `apps/expo-web/` |
| Corpus ambiguity analysis | ✅ measured | `scripts/analyze_corpus.py` |
| Error analysis | ✅ 12-class taxonomy | `scripts/error_analysis.py` |
| Augmentation ablation E1 | ⏳ running | `scripts/run_ablations.py` |
| Adhoc manual eval set | ⬜ needs human collection | `evals/adhoc_manual/` |

### The headline result

| System | Top-1 | CER | Top-5 | Size | ms / word |
|---|---|---|---|---|---|
| Rule engine (ours) | 9.2% | 48.8% | 13.6% | 3 KB | 0.10 |
| **Our model, int8, beam 4** | **53.4%** | **17.0%** | **64.6%** | **6.1 MB** | 27.4 |
| Our model, int8, greedy | 53.0% | 17.1% | - | 6.1 MB | **12.0** |

### The ablation

Two models, same architecture, same row count, same epochs, same seed. Only
the style of the extra rows differs.

| Training data | Standard test set | | Adhoc test set | |
|---|---|---|---|---|
| | Top-1 | CER | Top-1 | CER |
| standard rows only | **56.2%** | **15.4%** | 6.2% | 49.7% |
| standard + adhoc rows | 53.4% | 17.0% | **46.2%** | **20.0%** |

**Training on careful spelling alone makes a model three points better at
careful spelling and forty points worse at real typing.**

### What changed from the plan

1. **Dakshina is not the primary training data.** Aksharantar is, because it
   has 130 times more Malayalam pairs and a permissive licence. Dakshina's
   human-attested romanizations are still valuable, and the plan uses them
   where they are available.
2. **The model is a rules-propose, model-rank setup, not a standalone
   seq2seq.** Ezhuthola's rule engine turned out to be a strong baseline and
   a good candidate generator. See §4.7 and §7.2.
3. **The retroflex convention is barely used in the corpus.** Measured, not
   assumed: `T` appears in 6.6% of rows, `D` in 2.4%, `N` in 2.3%, and `R`
   and `S` never in capitals. Aligning the rule table with the corpus took
   the baseline from Top-1 6.1% to 9.2%.
4. **The decoder takes the whole prefix, not the last token.** A key value
   cache would be faster but complicates the exported graph.
5. **The corpus does not have to be held in memory.** The pipeline streams,
   because six and a half million pairs as python objects does not fit.
6. **The error analysis found a missing augmentation rule.** The model
   resolves the letter ambiguities well and gets the lengths wrong, so the
   long vowel rules were added after the first training run. They are off by
   default so that the pipeline reproduces the shipped checkpoint.
7. **The augmentation ablation is the result.** Standard-only training is
   three points better on standard input and forty points worse on adhoc
   input. The official validation set is in careful spelling and cannot show
   this, so a matching adhoc test set had to be built.
8. **The Kotlin runtime is built and tested, not just written.** The Gradle
   module compiles it and runs the real model through it.

---

## 1. Project at a glance

| | |
|---|---|
| **Problem** | Convert Manglish (Roman/Manglish) → മലയാളം. Harder than it looks: real users type in a style no public dataset covers. |
| **Contribution** | ① A systematic study of the **adhoc typing gap** (CER 7.4% → 22.7%). ② A **targeted augmentation pipeline** that closes it. ③ A compact **5.3M-param model** that fits in 5.4 MB. ④ A working **on-device Android demo** inside a real shipping keyboard. |
| **Model** | Char-level Transformer encoder-decoder, 4+4 layers, d=256, ffn=512, 4 heads (~5.3M params) |
| **Architecture** | 🟢 **Rules propose, the model disposes** (§7.2) — reuse Ezhuthola's 3 KB rule engine to generate ≤300 candidates; our model *ranks* them. One-function swap, airtight ablation. |
| **Data** | Dakshina (ML) + Aksharantar (ML) + our own augmentation → ~5.5M pairs |
| **Stack** | PyTorch → ONNX → ONNX Runtime (Web for the booth laptop, ORT Mobile for the phone) · Kotlin `InputMethodService` |
| **Cost** | **₹0** (Colab T4 is enough) to **₹2,000** (rented GPU + a ₹500 phone-swap contingency) |
| **Time** | **8 weeks** to expo-ready. 4-week compressed path in §14. |

### What you are claiming (write this on the poster)

> *Manglish → മലയാളം transliteration degrades by 3× on the typing style real users actually employ. We quantify the gap, build an augmentation pipeline that targets it, and ship a 5.4 MB on-device model that closes most of it — deployed as a drop-in replacement for the ranker inside [Ezhuthola](https://github.com/rag-creation/ezhuthola), a production Malayalam keyboard.*

That is a defensible, self-contained ML contribution. It has: a problem, a measurement, a method, an ablation, an artifact, and a demo.

### What you are *not* claiming

- ❌ "We built a better Manglish Keyboard than Manglish / Gboard" — you are not, and don't try.
- ❌ "New state of the art" — IndicXlit (MIT, 11M params) already exists and is your baseline.
- ❌ "We built a new rule engine" — Ezhuthola already has a good one. You build the *ranker*.
- ❌ "We translate Malayalam" — transliteration ≠ translation. Be careful with this word on the poster; a judge will notice.

---

## 2. The finding that makes this a project

The most important number in this entire document:

| Test set | Typing style | CER (Bi-LSTM + attention, published) |
|---|---|---|
| IndoNLP-2025 **Test Set-1** | standard, careful typing | **7.4%** |
| IndoNLP-2025 **Test Set-2** | adhoc — vowels dropped, gemination collapsed | **22.7%** |

*Source: Bajiyo et al., "Romanized to Native Malayalam Script Transliteration Using an Encoder-Decoder Framework", IndoNLP 2025 — `aclanthology.org/2025.indonlp-1.20`*

**A 3× CER degradation caused by nothing but *how people type*.** This is your project's thesis, and it is the reason a naive "train on Dakshina, get 7% CER, done" project is boring — and also why a naive one will disappoint users.

Your model must show the same cliff, then show that your augmentation closes it. That is a complete ML story:

```
Dakshina + Aksharantar, no augmentation   →  adhoc CER  ~20-24%   (reproduces the cliff)
        + our augmentation pipeline       →  adhoc CER  ~10-13%   (the contribution)
        + morphological post-processor    →  adhoc CER  ~9-11%    (bonus)
```

**If your augmentation does not move the adhoc number, the project fails.** That is your primary kill criterion (§8.4, Experiment E2).

### Why this matters for a demo

At the expo, type a word in careful spelling and then type it the way you actually would. If your model handles both, the audience sees it immediately. If it only handles careful spelling, the audience *also* sees it immediately — and you lose. This one demo beat explains your entire project better than ten slides.

---

## 3. Scope

### In scope
- Char-level transliteration model: Manglish → മലയാളം.
- Data pipeline: download → normalize → filter → augment → split.
- A **systematic evaluation**: 7 test sets, 4 metrics, 10 buckets, a 11-class error taxonomy.
- Ablations over the augmentation rules (this is the "experiment" part of the ML project).
- ONNX export + int8 quantization.
- **Web demo** (runs in the booth laptop's browser, fully offline).
- **Android demo** (an APK sideloaded onto your own phone — §11).
- A poster + a 3-minute pitch.

### Out of scope (and why)

| Cut | Reason |
|---|---|
| Reverse model (മലയാളം → Manglish) | Nice-to-have, not the thesis. Add only if you finish early. |
| Play Store release, AAB, policy, Data Safety | **You don't need it.** Sideload the APK. Saves ~2 weeks and all the legal overhead. |
| Voice typing, handwriting, stickers, emoji, themes | Not ML. Fork an existing keyboard (§11.1) and inherit them for free. |
| Full machine translation | Different problem, different models. Don't say "translate" in your title. |
| Multilingual (ml + ta + kn) | A v2 idea. Single language keeps the story clean. |
| 21-language Indic transliterator | IndicXlit already is one. |

### The cut list (if you fall behind)

Cut in this order. Each is worth roughly a week.

1. **Morphology post-processor** (§7.4) — 2–3 days, impressive but the model can partly compensate.
2. **CTC variant** (§5.2) — you probably won't need it.
3. **Android demo** — keep the web demo; show the model on the laptop. (Hardest to cut, so cut earliest.)
4. **Reverse model** — never start it.
5. **WebGPU backend** — WASM is enough for a booth laptop.
6. **int8 quantization** — fp16 at 10.6 MB still runs fine on a phone.

**Never cut:** the evaluation protocol (§9), the error analysis (§9.5), or the ablation table (§8.4). Those are what earn the marks.

---

## 4. Datasets

### 4.1 Dakshina (Google) — the brief's dataset, and it's good

| Property | Value |
|---|---|
| Repo | `github.com/google-research-datasets/dakshina` — **README only**; the data is in the tarball |
| Download | `https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar` (~9.16 GB, all 12 languages) |
| Malayalam path | `ml/` → `native_script_wikipedia/`, `lexicons/`, `romanized/` |
| Licence | **CC BY-SA 4.0** — fine for academic use. Attribute + share-alike the data. See §16. |
| Paper | Roark et al., LREC 2020 — `aclanthology.org/2020.lrec-1.294` |

**The exact files you need:**

| File | Contents | Count (ml) |
|---|---|---|
| `ml/lexicons/ml.translit.sampled.train.tsv` | `native \t romanization \t attestation_count` | 25,000 word types |
| `ml/lexicons/ml.translit.sampled.dev.tsv` | same | 2,500 word types |
| `ml/lexicons/ml.translit.sampled.test.tsv` | same | 2,500 word types |
| `ml/romanized/ml.romanized.rejoined.tsv` | `native \t roman` (unsplit sentences) | 10,000 strings |
| `ml/romanized/ml.romanized.split.tsv` | `native \t roman` (strings ≥30 chars segmented) | part of 10,000 |
| `ml/romanized/ml.romanized.rejoined.aligned.cased_nopunct.tsv` | token-aligned pairs + `</s>` | — |
| `ml/romanized/ml.romanized.split.validation.edits.txt` | per-line sub/del/ins counts | — |
| `ml/native_script_wikipedia/ml.wiki-filt.train.text.shuf.txt.gz` | native Wikipedia | 747.5K sentences |

**Why Dakshina is worth the download time (~9 GB):**
- Its romanizations were **elicited from native speakers** ("attested romanizations"). Aksharantar's *mined* rows were machine-generated and follow a stricter scheme. **Dakshina's roman forms are the closest thing to real Manglish in any public dataset.** This is exactly what you need for the adhoc story.
- The `attestation_count` column tells you how many annotators agreed on a romanization — use it as a confidence weight and to filter noisy forms.
- It's the canonical eval set. Every paper in this space reports on it, so you can compare directly.

> **Tip:** you don't need the whole 9 GB tarball. It's a plain `.tar` — you can stream it and extract only the `ml/` subtree:
> ```bash
> curl -L https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar \
>   | tar -x --wildcards --no-anchored 'ml/*'
> ```
> This avoids storing 9 GB. The `ml/` subtree is a small fraction of it.

### 4.2 Aksharantar (AI4Bharat) — volume

| Property | Value |
|---|---|
| Repo | `huggingface.co/datasets/ai4bharat/Aksharantar` — Malayalam slice `ml.zip`, ~125 MB |
| Volume (mal-en) | **~4,101K training pairs**; ~7.6K validation; ~12.5K test |
| Total corpus | 26M pairs / 20 languages |
| Format | JSONL: `uid`, native word, romanized word, `src` (`man`/`sam`/`wik`/`ind`/`exs`), `score` |
| Licence | *Manually collected* → **CC-BY**; *mined* → **CC0**; *existing sources* → **CC0** |
| Paper | Madhani et al., Findings of EMNLP 2023 — `aclanthology.org/2023.findings-emnlp.4` |
| Model | `ai4bharat/IndicXlit` (MIT) — see §4.4 |

**Why you need it:** Dakshina gives you ~30K unique Malayalam word types. Aksharantar gives you 4.1M pairs. Together, they let you (a) train a real model, and (b) show a clean ablation on *dataset size*.

**The `src` field is your best experimental lever.** Split your training data by `src`:
- `man` (manually collected) — small, human-written, closest to real typing.
- `sam`/`wik`/`ind` (mined) — large, machine-generated, standardized scheme.

Run the model on each subset separately. If performance differs a lot, **you have discovered a second, publishable observation**: that mined transliteration data teaches a different (more standard) distribution than human-typed data. That is a 3-line ablation and a great slide.

### 4.3 Evaluation-only — the north star

**IndoNLP-2025 shared task test sets** (accompanying Bajiyo et al., IndoNLP 2025):
- **Test Set-1** — general/standard typing patterns.
- **Test Set-2** — adhoc typing patterns, most vowel indicators missing.

Published Bi-LSTM baseline: **CER 7.4% / 22.7%**. **This is the benchmark your project is measured against.** Never train on it.

### 4.4 Evaluation-only — the baseline you must beat

**IndicXlit** (`ai4bharat/IndicXlit`, MIT, ~11M params) — a trained, ONNX-exportable transliterator covering 21 Indic languages.

| Model | Params | Dakshina-ml Top-1 | AK native Top-1 | AK NE Top-1 |
|---|---|---|---|---|
| **IndicXlit v1.0** | 11M | **63.56** | 64.73 | 33.93 |

**Run this in Week 1, before writing a single line of training code.** If your trained model doesn't beat it, you ship it and your project becomes "we benchmarked the ecosystem and found where the gap is" — which is *also* a valid project, but a weaker one. Knowing this number in Week 1 costs you 30 minutes and can save you 5 weeks.

**Also evaluate:**
- **🟢 The Ezhuthola rule engine — use it verbatim, do not build it.** This is your rule-based
  baseline and it already exists, in production, in 3 KB. See §4.7. This is the single most
  convincing comparison in your report, because the baseline is a *real shipping app*, not a toy
  your team wrote.
- **The Bi-LSTM + attention baseline** — Bajiyo et al.'s architecture. Reimplementing it is ~150 lines and gives you a same-family comparison. Strong for a college project.

### 4.7 Verified: Ezhuthola's engine is 100% rule-based — and it is a gift

**I checked the source rather than assuming.** `github.com/rag-creation/ezhuthola` contains **no
machine learning whatsoever.** Evidence:

| Check | Result |
|---|---|
| `gradle/libs.versions.toml` | Only AndroidX / Compose / JUnit. **No ONNX Runtime, no TensorFlow, no TFLite, no ML library.** |
| `app/src/main/assets/` | `ezhuthola_rules.json` (2,987 B) + word lists + fonts. **No `.onnx` / `.tflite` / `.pt` / model file.** |
| `CREDITS.md` | Credits word lists, Unicode data, CMUdict, fonts. **No model, no training data, no ML runtime.** |
| `licenses/` | Apache-2.0, CMUdict-BSD, ESDB-SCOWL, 4× OFL, Unicode-v3. **No ML runtime licence.** |
| APK size | **3.7 MiB** — too small to contain an ML runtime *plus* a model. |
| `tools/` | `build_emoji_data.py`, `make_en_malayalam.py` — data prep only, no training code. |
| `MalayalamEngine.kt` | Pure Kotlin (`org.json` only), hand-written longest-match parser + renderer. |

**What the rules file contains** (`ezhuthola_rules.json`) — this is the whole scheme in 3 KB:

| Section | Contents |
|---|---|
| `independent_vowels` / `vowel_signs` | `a/aa/A`, `i/ee/ii/I`, `u/oo/uu/U`, `e/E/ae`, `ai/ei`, `o/O/oa`, `au/ou` |
| `consonants` | 50 entries incl. `q`→ഖ, `Sh`/`shh`→ഷ, `zh`→ഴ, `L`→ള, `R`/`rr`→റ, `w`→വ, `f`→ഫ, `z`→സ |
| `clusters` | `nt`→ന്റ, `nk`→ങ്ക, `nch`→ഞ്ച, `ksh`→ക്ഷ, `x`→ക്സ, `nd`→ണ്ട, `mb`→മ്പ, `nth`→ന്ത, `sht`→ഷ്ട |
| **`alternatives`** | **The ambiguity inventory** — see below |
| `chillu_letters` | `n`→ൻ, `N`→ൺ, `r`/`R`→ർ, `l`→ൽ, `L`→ൾ, `m`→ം |
| `word_end_rules` | final `n/N/r/l/L` → chillu; final `m` → anuswaram; consonant+`u` may be virama |
| `consonants_mid_word` | `nj`→ഞ്ഞ (paranju) vs ഞ word-initially (njan); `d`→ട mid-word |
| `vowel_alternatives` | `e`→`E`, `o`→`O`, `a`→`aa` |

**The `alternatives` map is the definitive ambiguity list** — and it confirms the augmentation
rules in §5.2 were aimed at the right things:

```json
"c": ["ക","സ"],  "q": ["ഖ","ക"],  "t": ["ട","ത","റ്റ"],
"th": ["ത","ഥ"], "tt": ["ട്ട","റ്റ","ത്ത"],  "d": ["ദ","ട","ഡ"],
"n": ["ന","ണ"], "nn": ["ന്ന","ണ്ണ"],  "l": ["ല","ള"],
"ll": ["ല്ല","ള്ള"], "r": ["ര","റ"],  "s": ["സ","ശ","ഷ"],
"sh": ["ശ","ഷ"], "ng": ["ങ്ങ","ങ"]
```

**Three things this changes:**

1. **Don't build a rule baseline. Use this one.** It is better than anything you would write, and
   it is already a fair, published, shipping baseline.
2. **Adopt the rules-propose / model-rank architecture** (§7.2). It is less code, faster, more
   accurate on orthography, and gives a clean one-function ablation.
3. **The gap you attack is now precisely defined:** a rule engine's ranker is a *frequency list*.
   It has **no context, no notion of adhoc typing, and no ability to score an unseen word.**
   That is exactly the ceiling a learned model lifts. Your poster can say it in one line:
   *"Rules propose 300 spellings; a frequency list picks one. We replace the picker."*

**Also read, and reuse rather than rebuild:**

| File | Why it matters |
|---|---|
| `engine/MalayalamEngine.kt` | The rule engine. Parse → render → expand. Your candidate generator. |
| `engine/SuffixSplitter.kt` | **This is the "M1 morphology module" from §7.4, already implemented** — with the real sandhi rules: ം-noun `ത്ത` gemination, chillu+vowel, `y`/`v` glides, `യ്` insertion before `ക്ക്` after an അ-ending, and the `ATH` variants (`athth`/`atth`/`ath`). Don't rebuild it. |
| `engine/MalayalamSuggester.kt` | The frequency-list ranker you are replacing. |
| `CREDITS.md` | Confirms the chillu normalisation advice (§5.1): *"old-style chillu letters converted to atomic chillus (ന് → ൻ at word end)"*. Independently validates Trap 1. |

**Bonus finding — the rules encode real-world hacks that training data won't teach you:**

- `softC()`: `c` before `e`/`i`/`y` → സ (cinema → സിനിമ, city → സിറ്റി) — English-influenced.
- `normalizeAutoCapital()`: phones auto-capitalise the first letter, so `Njan` must mean ഞാൻ,
  not ണ + ജാൻ. A model trained on clean data gets this wrong; the rule engine gets it right.
- `u` is genuinely ambiguous: `kandu` → കണ്ടു but `veedu` → വീട്. The engine emits **both** and
  lets the ranker decide. That is precisely the "generate alternatives, rank them" pattern.

### 4.5 Evaluation-only — Mozhi conformance

**Mozhi** is the de-facto Manglish romanization standard for Malayalam input, used by Google Input Tools and Indic Keyboard's "Transliteration (Mozhi)" layout. Its design goals are explicitly *"Naturalness — close to transliteration traditions of Malayalees"*, *"Consistency"*, *"Economical about keystrokes"*, *"Shift-free"*.

The Mozhi 2.0 spec publishes a **rule-set and test cases** (`mozhi-documentation.digitalmalayali.in`). Turn those into an automated conformance suite. Your model should agree with the rule engine wherever the scheme is unambiguous.

```
Vowels    a  aa  i  ii  u  uu  R  RR  e  E  ai  o  O  au  Ll  Lll  am  aH
Cons.     k  kh  g  gh  ng   |  ch chh j jh nj   |  T Th D Dh N
          th thh d dh n       |  p ph b bh m       |  y r l v
          S sh s h            |  L zh rr t
Chillus   r(ർ)  n(ൻ)  N(ൺ)  l(ൽ)  L(ൾ)
```

### 4.6 Supplementary — the adhoc ground truth

Your own hand-collected set. **This is the most valuable dataset you will create and it takes one afternoon.**

- Ask 10–15 people (or just yourself, typing the way you actually type) to write 50–100 words in Manglish.
- Record the Malayalam you *meant*.
- Store as `roman \t native \t style` with `style ∈ {standard, adhoc}`.
- Target: **500–1,000 words** for the demo, scale to 2,000 if you can.

This gives you a **qualitative** eval set that no public dataset matches, and it is a legitimate research contribution for a college project. It also makes the poster's headline chart yours rather than someone else's.

---

## 5. Data pipeline

```
Dakshina ml/            Aksharantar mal/        Your collected set
(lexicon + romanized)   (4.1M pairs)            (500–2,000 adhoc)
        │                     │                        │
        └─────────────┬───────┴────────────────────────┘
                      ▼
            ┌───────────────────────┐
            │ 1. normalize.py       │  NFC · chillu legacy→atomic ·
            │                       │  ZWJ handling · case preservation
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 2. filter.py          │  length · mixed-script ·
            │                       │  dedupe by native word · identity
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 3. augment.py         │  15 rules targeting adhoc style
            │    (the contribution) │  each row carries a `style` tag
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 4. build_splits.py    │  split by NATIVE WORD (no leakage)
            │    + dictionary build │  trie for the fast path
            └───────────────────────┘
```

### 5.1 Normalization — the Malayalam-specific traps

Get these wrong and you ship invisible broken characters that render as boxes on some devices. These are the details that make the project look *serious*.

**Trap 1 — Chillu dual encoding.** Malayalam's vowel-less "chillu" forms have two Unicode representations:

| Form | Example | Where it comes from |
|---|---|---|
| Legacy (Unicode 5.0) | `ണ` + `്` + `ം` (consonant + virama + ZWJ) | 2019-era Wikipedia → **present in Dakshina** |
| **Atomic (preferred)** | `ൺ` U+0D7A | `ൻ` U+0D7B, `ർ` U+0D7C, `ൽ` U+0D7D, `ൾ` U+0D7E, `ൿ` U+0D7F, plus historic `ൔ` U+0D54 / `ൕ` U+0D55 / `ൖ` U+0D56 (Unicode 9.0, 2016) |

Unicode's own guidance: *"applications should also be prepared to handle data in the representation specified in Unicode 5.0 … while **an input method should output standard chillus**."* → **Always output atomic chillus.** You must map legacy sequences explicitly; **NFC does not do this for you.**

**Trap 2 — ZWJ/ZWNJ are semantic.** `ന`+`്`+ZWJ (chillu-n, "avan") vs `ന`+`്`+ZWNJ (explicit virama, "avanu") are *different words*. Never strip joiners blindly — strip them only as part of the legacy→atomic chillu mapping.

**Trap 3 — Don't use NFKC.** It destroys the chillu-vs-virama distinction. Use **NFC** only (which handles canonical reordering of Malayalam vowel signs like ൊ ോ ൌ ൗ).

**Trap 4 — Roman case carries meaning.** In Mozhi, `T`/`Th`/`D`/`Dh`/`N` are retroflex ട/ഠ/ഡ/ഢ/ണ; `t`/`th`/`d`/`dh`/`n` are dental ത/ഥ/ദ/ധ/ന. Uppercase is *meaningful*. Manglish users rarely press Shift, so:
- Keep case in the source (don't lowercase unconditionally — it destroys retroflex info).
- **Augment with case-folded variants** so the model also handles lowercase-only typing.
- The retroflex/dental confusion is a *feature* of the problem, not a bug to normalize away.

**Trap 5 — The target vocab is not "Malayalam letters".** It includes 15 independent vowels, 36 consonants, ~15 dependent vowel signs, virama (chandrakkala), anusvara, candrabindu, 9 chillus, the historic dot-reph (ൎ) and para sign (൥), plus digits. Build the vocab from the actual training data.

### 5.2 Augmentation — the core contribution

This is where your project lives. The rules below are derived from how Malayalis actually type, from Mozhi's "Naturalness" goal, and from the observed CER cliff.

| # | Rule | Example | Why |
|---|---|---|---|
| A1 | **Inherent-vowel drop** | `sundari` → `sundr`; `kerala` → `keral` | Malayalam's inherent `a` is often not typed. **The single largest cause of the adhoc CER blowup.** |
| A2 | **Gemination collapse** | `katt` → `kat`; `pannan` → `panan` | Modern Malayalam mostly doesn't geminate; typers write one consonant. |
| A3 | **Final-consonant drop** | `keralathil` → `keralathl` | Fast typers drop it. |
| A4 | **Chillu substitution** | `ൽ`→`l`; `ർ`→`r`; `ൻ`→`n`; `ൺ`→`N`/`nn`; `ം`→`m` | Manglish chillus are usually written as the base consonant. |
| A5 | **Aspirate drop** | `kha`→`ka`; `gha`→`ga`; `thha`→`tha` | Northern/Malabar accent differences. |
| A6 | **Aspirate add** | `ka`→`kha`; `sa`→`sha` | The `sa`/`sha` and `cha`/`chha` ambiguity. |
| A7 | **Retroflex↔dental confusion** | `tha`↔`Ta`; `na`↔`Na`; `da`↔`Da` | The `T`/`t` distinction is invisible to most typers. Apply case-flip. |
| A8 | **Palatal simplification** | `ch`/`chh`→`c`/`ch`; `nj`→`n` | — |
| A9 | **ഴ / ള variants** | `ഴ`→`zh`/`z`/`r`; `ള`→`l`/`zh` | ഴ is the highest-friction letter in Malayalam; writers use `zh`, `z`, or `r`. |
| A10 | **ശ / ഷ / സ merge** | `ശ`→`sh`, `ഷ`→`sh`, `സ`→`s`/`sh` | The largest ambiguity class in the language. |
| A11 | **Nasal-cluster reduction** | `ന്ത`→`nth`/`nt`; `ന്റ`→`nth`/`nd` | — |
| A12 | **Compound joining** | `ente peru` → `enteperu` | Manglish compounds are often typed without spaces. Needs segmentation — §7.3. |
| A13 | **Mozhi synonym injection** | `za`→`ശ`; `Sa`→`ശ`; `rr`→`റ`; `t`→`റ` | Teach the model Mozhi's alternate spellings so it prefers Mozhi-canonical output. |
| A14 | **Latin-digit noise** | `1` for `൧`, `5` for `൫` | Malayalam numerals are ASCII digits in practice. |
| A15 | **Word-boundary noise** | random space insert/remove | Robustness for partial input. |

**Augmentation budget:** apply A1–A15 with per-rule probabilities (A1 high ~0.35, A4 high ~0.5, A10 medium) to generate ~1.5–2.5M additional pairs.

**Every augmented row carries a `style` tag** (`standard` / `adhoc` / `mixed`). This is what makes your ablation table possible — you can train on subsets of rules and measure each rule's contribution. **That ablation table is your best slide.**

### 5.3 Splits

| Split | Source | Size | Purpose |
|---|---|---|---|
| `train` | Dakshina lexicon + Aksharantar train + augmented | ~5.5M | Training |
| `valid` | Aksharantar valid | 7.6K | Early stopping |
| `test-standard` | Aksharantar test | 12.5K | Held-out general |
| `test-adhoc` | **IndoNLP-2025 Test Set-2** | — | 🔴 **North star** |
| `test-standard-ext` | IndoNLP-2025 Test Set-1 | — | 🟠 Control |
| `test-dakshina` | Dakshina ml dev + test lexicon | 5,000 | 🟠 Head-to-head with published numbers |
| `test-ne` | AK-NEF + AK-NEI (ml) | 835 + 1,219 | 🟡 Named entities (hard) |
| `test-mozhi` | Mozhi spec test cases | hundreds | 🟡 Scheme conformance |
| `test-manual` | Your hand-collected adhoc set | 500–2,000 | 🟡 Ground truth for style |

**Leakage control — critical:** split by **native word**, not by romanization. Otherwise `sundari` (train) and `sundri` (test) leak, and every number you report is inflated. Dakshina already partitions by native-script word with dev/test lemmata unseen in train — mirror that discipline for Aksharantar.

**Dedupe:** collapse duplicate `roman → native` pairs. Keep `attestation_count` as a confidence weight and as a tie-break prior.

---

## 6. Model

### 6.1 Why character-level

- Malayalam is a **matras-heavy abugida**: one grapheme is 2–4 codepoints (consonant + virama + vowel sign). Subword BPE tokenizers split graphemes — exactly where transliteration models break.
- **No OOV.** A keyboard must handle any string, including names, slang, URLs, code. Char-level never corrupts unknown input.
- Tiny vocab (≈45 source + ≈80 target) → the embedding table is negligible → the model can be small.

### 6.2 Config — `ml-mini`

| Hyperparameter | Value | Rationale |
|---|---|---|
| Architecture | Transformer enc-dec, pre-norm | Matches IndicXlit's proven recipe (`--layernorm-embedding`, `--decoder-normalize-before`) |
| Encoder layers | 4 | IndicXlit uses 6; 4 suffices for one language pair |
| Decoder layers | 4 | — |
| `d_model` | 256 | Same as IndicXlit |
| FFN dim | 512 | IndicXlit uses 1024; 512 is enough for a 45-char vocab |
| Attention heads | 4 | — |
| Dropout | 0.3 | — |
| Activation | GELU | — |
| Source vocab | 45 | `a-z`, `0-9`, `'`, `-`, `.`, `,`, `/`, space, PAD/BOS/EOS |
| Target vocab | ~80 | Derived from training data (§5.1 Trap 5) |
| `max_src_len` | 48 | Longest realistic Manglish word |
| `max_tgt_len` | 64 | Malayalam output is longer than input (matras) |
| Embedding tying | Off (different scripts) | — |

**Parameter budget (worked out):**

```
Embeddings       (45 + 80) × 256            =  32,000
Encoder layer    attn 4×(256×256)+bias      = 263,168
                 ffn  2×(256×512)+bias      = 263,680
                 LN × 2                     =   1,024
                 → 527,872 × 4 layers        = 2,111,488
Decoder layer    self-attn                  = 263,168
                 cross-attn                 = 263,168
                 ffn                        = 263,680
                 LN × 3                     =   1,536
                 → 791,552 × 4 layers        = 3,166,208
Output proj      80 × 256 + bias            =  20,560
Final LN                                      =     512
────────────────────────────────────────────────────────
TOTAL                                       ≈ 5,330,768  (5.3M)
```

| Precision | Size |
|---|---|
| fp32 | 21.3 MB |
| fp16 | 10.6 MB |
| **int8 (dynamic, weights)** | **~5.4 MB** ← the demo artifact |

For reference, IndicXlit is 6+6 layers with ffn=1024 ≈ **11M params** — `ml-mini` is less than half the size, and trained on Malayalam-only data.

### 6.3 Optional CTC variant

| Property | Value |
|---|---|
| Architecture | 6-layer encoder + CTC head (no decoder) |
| Params | ~4.8M |
| Latency | ~5–10 ms/word (single forward pass) |
| Weakness | CTC's monotonic alignment is a liability when the roman→native mapping is non-monotonic (`sh` → 2 target chars, and vice versa) |

**Recommendation: build seq2seq only.** If you need a per-keystroke live preview, a **rule-based Mozhi prefix engine** is faster *and* more accurate than neural CTC. Add CTC only if you have spare time — it is a good "we tried an alternative architecture" ablation for the report.

### 6.4 Reverse model (stretch goal)

മലയാളം → Manglish. Same architecture, source/target swapped, same pairs. **Only if you finish 2 weeks early.** It needs a language-detection gate so you don't transliterate English text.

Bonus: **round-trip consistency** (`roman → native → roman == roman`) is a free correctness metric for the forward model.

---

## 7. Decoding, morphology, and segmentation

### 7.1 Decoding

| Setting | Value | Where |
|---|---|---|
| Greedy | beam=1 | Live preview, latency-critical path |
| **Beam search** | **beam = 4–8** | Commit path. IndicXlit uses beam 4; the community ONNX port reports ~6% gain from beam-4 + rescoring |
| n-best | top 5 | Feeds the suggestion strip |
| Rescoring | blend model log-prob with a Malayalam word-frequency prior, α ≈ 0.9 model / 0.1 dict | Resolves `sughamano` vs `sukhamano` |
| Length penalty | ~1.0 | Avoids truncation bias on long Malayalam output |

### 7.2 The recommended architecture — rules propose, the model disposes

> ⚠️ **This section was rewritten after verifying Ezhuthola's source.** The original plan proposed a
> three-tier pipeline (dictionary → morphology → neural). Reading Ezhuthola's engine showed a
> **better** architecture that we should copy. See §4.7 for the verification.

Ezhuthola's `MalayalamEngine.transliterate()` works in three steps:

1. **Parse** the Manglish into segments using longest-match against a rule table
   (`ezhuthola_rules.json`, 3 KB).
2. **Render** with position-aware orthography: virama before a following consonant, chillu at word
   end, anuswaram for a final `m`, `nj` → ഞ്ഞ mid-word but ഞ word-initially, `u` → virama vs. `u`
   vowel depending on context.
3. **Expand** — generate up to **300 candidates** by applying one swap (ന↔ണ, ല↔ള, ശ↔ഷ, െ↔േ,
   ട↔ത↔റ്റ, …) and then two swaps at different positions.

Those 300 candidates are then **ranked by a frequency list**. That last step is where a learned
model earns its keep.

```
user types "sneham"
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  RULE ENGINE  (Ezhuthola's, 3 KB of JSON)   │  ← reuse verbatim, don't rebuild
  │  parse → render → expand to ≤300 candidates │     (e.g. സ്നേഹം, ശ്നേഹം, ഷ്നേഹം, സ്നേഹാം, …)
  └────────────────────┬────────────────────────┘
                       │  candidate set
                       ▼
  ┌─────────────────────────────────────────────┐
  │  LEARNED RANKER  (our seq2seq / scorer)    │  ← the ML contribution
  │  P(Malayalam candidate | Manglish)          │
  │  + frequency prior + context                │
  └────────────────────┬────────────────────────┘
                       ▼
            top-5 → suggestion strip
```

**Why this beats a from-scratch seq2seq:**

| | From-scratch seq2seq | Rules → neural rank |
|---|---|---|
| Context-sensitive orthography (virama, chillu, anuswaram, `nj`→ഞ്ഞ) | Must be *learned* from data; commonly wrong | Handled **deterministically and correctly** |
| Ambiguity (ശ/ഷ/സ) | Must be learned | Rules **enumerate** the options; model **picks** |
| Search space | Beam over the full output vocabulary | Score a bounded set of ≤300 well-formed candidates |
| Speed | Encoder + N decoder steps | One encoder pass + cheap scoring |
| Explainability | Opaque | "The rule said these 300 are possible; the model ranked them 1–5" |
| Ablation | Needs a from-scratch rule engine to compare against | **Swap one function** — `rank(candidates)` — and diff the outputs |

**The last row is the important one.** Because the candidate generator is *identical*, any
difference in output is attributable purely to the ranker. That is a clean, airtight ablation:
*"same candidates, different ranker, and here is exactly what the learned ranker buys."*

**This also reframes the project's contribution, for the better:**

> *Ezhuthola's rule engine proposes up to 300 candidate spellings per word and picks one with a
> frequency list. That ranker cannot generalise: it has no notion of context, no notion of how
> people actually type, and no ability to score an unseen word. We replace it with a learned
> char-level model and quantify the gain.*

That is a *much* better ML project than "we trained a seq2seq from scratch," because:
- The baseline is a **real, shipping, zero-permission app** — not a toy you wrote.
- The comparison is apples-to-apples.
- The demo is a **real app you can install next to yours** and type into.
- You are making a **genuine contribution to an open-source project**, which is a great thing to
  have on a poster.

**The fallback (if the rules-propose architecture proves fiddly):** keep the original three-tier
design — dictionary → morphology → neural beam. It works. But try the rule-based version first;
it is less code and a better story.

### 7.3 Word segmentation (for the translator use case)

Manglish compounds are often typed without spaces: `entepremangalicheyil` for "ente premangalicheyil". A word-level model can't handle that.

**Simple approach:** try the whole string as one word, and try every split point; score all candidates with the model + a word-boundary prior; pick the best. **Better approach:** train a tiny char-level BiLSTM segmenter (~1M params) on Dakshina's `*.aligned.cased_nopunct.tsv` files, which give token-aligned pairs for free.

**For the keyboard demo this is unnecessary** — the user pressed space. For a bulk-text translator it is essential. Scope it as a stretch goal.

### 7.4 Morphology — `keralathil` → കേരളത്തിൽ

A pure seq2seq produces `കേരളതിൽ` here. It has no idea that Malayalam postpositions **double the preceding consonant** when they start with `t`. This is **morphology, not transliteration**, and it's the most visible failure a user will spot.

| Option | Approach | Effort | Quality |
|---|---|---|---|
| **M1 (recommended)** | **Suffix lexicon + sandhi.** Strip a known postposition (`-il`, `-ude`, `-inte`, `-e`, `-um`, `-aayi`, `-ayirunnu`, `-aakki`…) from the roman token, transliterate the stem, apply the gemination rule. | 2–3 days | Very high on common cases |
| M2 | Augment training data with `stem+suffix` pairs generated from a suffix lexicon | ~1 day + lexicon | High, given a good lexicon |
| M3 | Full morphological analyzer (SMC's `mlmorph`) | Weeks | Overkill |

**Do M1 + M2.** It's a *visible* demo win and a clean ablation row.

**Related:** English words in Manglish mode get Malayali orthography — `four` → ഫോർ, `station` → സ്റ്റേഷൻ. That needs a CMU Pronouncing Dictionary-driven phoneme→Malayalam mapper (CMUdict is BSD-licensed). Nice stretch goal.

---

## 8. Training

### 8.1 Recipe

| Item | Value |
|---|---|
| Framework | PyTorch 2.x |
| Loss | Label-smoothed cross-entropy (ε = 0.1) |
| Optimizer | AdamW, β = (0.9, 0.98), weight decay 0.01 |
| LR | 7e-4 peak, inverse-sqrt or cosine with warmup |
| Warmup | 4,000 steps |
| Batch | 1,024 sentences, length-bucketed |
| Epochs | 20–40, early stop on `valid` CER |
| Precision | bf16 AMP on GPU, fp32 master weights |
| Checkpoint selection | Lowest `test-standard` CER **subject to** `test-adhoc` CER not regressing >2 points — this guards against overfitting the standard distribution and is exactly the kind of detail that makes an ML project look rigorous |
| Seeds | 3 for the final run; report mean ± std |

### 8.2 Hardware

| Stage | Requirement |
|---|---|
| Training (5.3M model, 5.5M pairs, ~20 epochs) | Single **RTX 4090 / A100** ≈ 4–8 h, or a **T4 / L4** ≈ 1–2 days. **Colab / Kaggle free T4 is sufficient.** |
| Ablations + 3 seeds | Same, ×3–4 runs. Still free-tier territory. |
| Export / quantize / eval | CPU-only laptop, minutes |

**This is not a project that needs GPU money.** Colab Pro (~₹900/month) is a comfortable budget if you want a faster T4/V100.

### 8.3 Efficiency tricks (save hours)

- **Length-bucketed batching** — Malayalam targets are ~1.5–2× the source length; bucketing saves >40% of compute.
- **Char-level means tiny sequences** (max 48 tokens) — no need for memory-efficient attention.
- **`torch.compile`** — easy 1.5–2× on the training loop.
- **Checkpoint every epoch, keep top-5** — cheap, lets you pick post-hoc.

### 8.4 Experiment ladder — and the kill criteria

Run in order. Each has a **kill criterion**. This table becomes your report's results section.

| # | Experiment | Purpose | Kill criterion |
|---|---|---|---|
| **E0** | **Baselines**: IndicXlit, **Ezhuthola's rule engine** (use their code, §4.7), (optionally) Bi-LSTM+attn on all eval sets | The numbers you must beat | — |
| **E1** | `ml-mini` on Dakshina + Aksharantar, **no augmentation** | Reproduce the adhoc cliff | — |
| **E2** | E1 + full augmentation pipeline | **The main hypothesis** | **If adhoc CER improves < 3 points absolute, the augmentation design is wrong → go to E3 before E4** |
| **E3** | E2 + chillu / case / ZWJ normalization fixes | Verify each normalization matters | Each fix must not regress standard CER > 0.5 |
| **E4** | **Per-rule ablation**: train with each augmentation rule removed | *Which rules matter?* | — (this is your best slide) |
| **E5** | E2 + frequency rescoring at decode | Add a Malayalam n-gram prior | Report raw **and** rescored |
| **E5b** | **🟢 Rules-propose / model-rank** (§7.2): Ezhuthola's rule engine generates candidates, our model ranks them | **The recommended architecture** | If the model-ranker beats the frequency ranker on the *same* candidate set, you have a clean win |
| **E6** | Train on `src=man` only vs. mined only vs. mixed | **Does human-typed data matter?** | — (second-best slide) |
| **E7** | + morphology post-processor | Fix `-thil/-inte/-ude` | — |
| **E8** | CTC variant (optional) | Alternative architecture | — |
| **E9** | int8 quantization | The shipped artifact | Report the accuracy delta |

**The E4 and E6 ablations are what turn this from "we trained a model" into a real ML project.** Budget time for them.

---

## 9. Evaluation

Full protocol in [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md). Summary:

### 9.1 Metrics

| Metric | Definition | Why |
|---|---|---|
| **CER** | `Σ Levenshtein / Σ len(ref)`, micro-averaged | Primary. Standard in this literature. |
| **WER** | Word-level error rate | Secondary. |
| **Top-1** | `fraction(hyp == ref)` exact match | **The headline number** — this is what Dakshina/IndicXlit report. |
| **Top-5 recall** | `fraction(ref ∈ top5)` | What matters for a *suggestion strip* — the user picks from a list. |
| **Round-trip** | `roman → native → roman == roman` | Free correctness proxy. |
| **Latency** | p50/p95 ms per word on target hardware | Product metric. |
| **Size** | MB | Product metric. |

> ⚠️ **Compute Levenshtein on grapheme clusters, not codepoints.** Malayalam's vowel signs and virama are combining marks; counting them separately makes CER meaningless and makes two different models look identical. Use `regex` with the `\X` flag, or the `grapheme` package.

### 9.2 Report format

```
set                  n       Top-1   Top-5    CER      WER     RT-acc
test-standard        12500   0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-adhoc           N       0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-dakshina        5000    0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-ne              2054    0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
test-manual          N       0.XXX   0.XXX    0.0XX    0.0XX   0.XXX
```

**Mandatory companion rows:** IndicXlit baseline · rule-based (Mozhi) baseline · ours-int8. Always all three, always from the same harness.

### 9.3 Bucketed breakdown (aggregate CER hides the failures that matter)

| Bucket | Why |
|---|---|
| Input length (1–3 / 4–6 / 7–9 / 10+) | Long words accumulate error |
| Frequency (head / mid / tail) | Memorisation vs. generalisation |
| **In-vocab vs. OOV** | 🔴 **The most important one.** In the shipped product, in-vocab words hit the dictionary and never reach the model — so the model's real-world quality ≈ its **OOV** performance. Report OOV separately. |
| Contains chillu | Surfaces Unicode-normalisation failures |
| Contains geminate | Sandhi errors |
| Contains cluster | Conjunct formation errors |
| Contains retroflex | The `T`/`t` case distinction |
| Postposition suffix | Tests the morphology module |
| Latin loanword | `four` → ഫോർ class |

### 9.4 Error taxonomy

Dump the top-50 worst errors per run to `evals/results/errors/<run_id>.tsv` and manually classify ≥100 into a fixed taxonomy:

`vowel-drop` · `gemination` · `chillu` · `cluster` · `retroflex` · `sh-vs-s` · `zh-vs-l` · `morphology` · `segmentation` · `ne-unknown` · `latin-leak`

**If a class accounts for >20% of errors, it gets a targeted data fix in the next iteration.** This closes the loop between evaluation and augmentation.

### 9.5 Reproducibility

Every run records: git SHA, config, checkpoint SHA256, dataset hash, tokenizer version, seed, library versions. `evals/results/eval.json` is committed and regenerated by CI; the web demo reads it directly, so published numbers are always the committed numbers.

---

## 10. The demo (expo deliverable)

Two demos. Build the web one first — it is the safe path.

### 10.1 Web demo (primary — build this first)

A single-page app that runs the model **entirely in the browser**. No server, no network, works on the booth laptop with the wifi off.

| Feature | Detail |
|---|---|
| **Live transliteration** | Type Manglish → Malayalam, updating per keystroke (debounced 80 ms, in a Web Worker) |
| **🔴 The killer feature: style toggle** | Toggle *careful spelling* ↔ *how people actually type*. The model's output quality visibly changes. **This single toggle demonstrates your entire project.** |
| **Rule vs. model diff** | Side-by-side Mozhi-rule output vs. model output, with a character-level diff. Shows where the ML adds value. |
| **Live CER readout** | Running accuracy on the current session + the full benchmark table from `eval.json` |
| **Head-to-head** | IndicXlit vs. ours on the same input, with latency and size. Honest comparison. |
| **Test-suite runner** | Run the adhoc / standard / Mozhi suites in-browser and watch the numbers |
| **Offline badge** | "Runs 100% in your browser. No server. Nothing leaves this page." |

**Stack:** Next.js 15 (App Router) + React 19 + TypeScript, static-exportable · **ONNX Runtime Web** (WASM + SIMD; optional WebGPU) · Tailwind + shadcn/ui · **Noto Sans Malayalam** (SIL OFL) for rendering · GitHub Pages / Cloudflare Pages.

**Effort:** ~9 days.

| Task | Days |
|---|---|
| ONNX export + quantize + web smoke test | 1 |
| Next.js scaffold + worker + tokenizer | 2 |
| UI: input/output panes, style toggle | 2 |
| Benchmark panel + `eval.json` generation | 1 |
| Head-to-head vs. IndicXlit | 1 |
| Polish, responsive, a11y, offline PWA | 2 |

### 10.2 Android demo (secondary — the "wow" artifact)

Proves the model runs **on-device with zero permissions**. Sideload an APK; no Play Store involved.

**What to build:** a minimal `InputMethodService` that:
- Provides a QWERTY key row (or reuses the host keyboard's hardware input).
- On word boundary, runs the model and shows the top-5 Malayalam candidates in a suggestion strip.
- Tapping a candidate commits it.
- Shows the model size and inference time in a debug overlay — *that's* the on-device proof.

**What to skip:** themes, emoji, stickers, settings screen polish, voice. Those are not ML and they are what turns a 3-week task into a 3-month one.

### 10.3 What to show at the booth (a 3-minute script)

1. **30 s** — Type `ente peru` → എന്റെ പേര്. "This is what Gboard does."
2. **60 s** — Toggle **adhoc mode**. Type `ente peru` again, but as you'd actually type it — vowel-dropped, gemination collapsed, no caps. Show the model still gets it, and show the **rule engine's** output failing on the same input. **This is the demo.**
3. **30 s** — Show the benchmark table: your model vs. IndicXlit vs. rules, on standard *and* adhoc.
4. **30 s** — Show the error taxonomy: the top-3 error classes from your analysis, and which augmentation rule fixed each.
5. **30 s** — Phone: the same model running on-device, zero permissions, 5.4 MB, X ms/word.

---

## 11. Android — the difficulty question, answered for a college project

> *"How difficult would it be to implement this as an alternative to, say, Manglish Keyboard on Android?"*

**Short answer: a demo-quality Manglish IME is 2–3 weeks. Matching Manglish is still 4–6 months, and you should not attempt it.** For a college project, the question is really "can I show my model running in a real Android keyboard?" — and yes, in about a week if you do it smartly.

### 11.1 Layered difficulty

| Layer | What | Difficulty | 1 student |
|---|---|---|---|
| Data pipeline | normalize, augment, split | Medium | 4–6 d |
| Train the model | train, tune, evaluate | Medium | 5–8 d |
| ONNX export + quantize | export, parity test, int8 | Low–Med | 1–2 d |
| Cross-platform inference | JS + Kotlin, identical outputs | Medium | 3–4 d |
| **Minimal IME shell** | `InputMethodService` + suggestion strip | **Med–High** | **7–12 d** |
| **Fork an existing keyboard instead** | swap the model into a real app | **Low** | **1–3 d** ⭐ |
| Polish (themes, settings, a11y) | | Medium | 2–3 wk |
| Voice / handwriting / stickers | | Very high | 2–3 mo |
| **Feature parity with Manglish** | | **Very high** | **4–6 mo** |

### 11.2 The shortcut: fork an existing keyboard

Two mature open-source shells exist. **Do not write an `InputMethodService` from scratch for a college project.**

| Option | Licence | Stack | Malayalam support | Notes |
|---|---|---|---|---|
| **Ezhuthola** (`com.rr.numio.ezhuthola`) | **GPL-3.0** | Kotlin + Jetpack Compose | Manglish keyboard **already built**; **rule-based** transliteration engine | 3.7 MiB, **zero permissions**, suggestions + morphology + stickers + emoji + clipboard + themes. **Closest to the goal.** ⭐ |
| **Indic Keyboard** (SMC) | **Apache-2.0** | Java (+JNI) | Transliteration (Mozhi), Phonetic, Inscript, Swanalekha, Mobile Inscript | Full-featured, 23 languages / 54 layouts, AOSP heritage. |

**The play — and it is now much cheaper than it looks.** Because Ezhuthola's engine is a pure
rule engine with no ML (§4.7), swapping in a model is a **one-function change**:

```
Ezhuthola:  MalayalamEngine.transliterate()  →  300 candidates  →  MalayalamSuggester ranks
Your fork:  MalayalamEngine.transliterate()  →  300 candidates  →  YOUR model ranks  ← 1 function
```

You keep the parser, the renderer, the candidate expander, the morphology splitter, the keyboard
UI, themes, emoji, stickers, clipboard, user-word learning. You replace **one ranking function**
with your ONNX model.

**One to three days of work, and the demo looks better than anything you'd build** — because it
already has themes, emoji, and stickers. You also get to say on your poster that you contributed
a better ranker to a real open-source keyboard with real users. That is a strong line.

> ⚠️ **Licence note:** Ezhuthola is GPL-3.0, so your fork is GPL-3.0. For a college academic project that is completely fine — just release under GPL-3.0 (or keep the fork private for the demo). If you ever wanted to publish a closed-source app, use Indic Keyboard (Apache-2.0) instead. **GPL-3.0 does not prevent you from using a Dakshina-trained model** — CC BY-SA and GPL-3.0 are compatible for academic distribution.

### 11.3 The hard parts, honestly

**Model-side (the interesting problems):**
1. **Adhoc typing style** — the 7.4% → 22.7% cliff. Fixable, but it *is* the project.
2. **ശ / ഷ / സ ambiguity** — `sh` → ശ or ഷ; `s` → സ or ശ. Nearly unresolvable from the roman form alone. Needs corpus statistics + context.
3. **Morphology, not transliteration** — `keralathil` → കേരളത്തിൽ needs a separate module (§7.4). **Ezhuthola already implements this** in `SuffixSplitter.kt` — read it, don't rebuild it.
4. **Word segmentation** in compound typing (§7.3).
5. **Named entities are hard** — IndicXlit scores only ~34% on AK-NEF/NEI. And names are exactly what people type.
6. **Chillu and Unicode normalization** (§5.1) — invisible until wrong, then catastrophic (tofu boxes). Ezhuthola's `CREDITS.md` confirms the fix independently.
7. **🟢 The ranker's ceiling (the one that matters).** Ezhuthola ranks its 300 candidates with a **frequency list**. A frequency list has no context, no notion of adhoc typing, and cannot score an unseen word. **This is precisely the gap your model fills** — and it is the sharpest problem statement you can write.

**Shell-side (the tedious problems):**
7. **`android.inputmethodservice.Keyboard` / `KeyboardView` are deprecated (API 29).** You must write your own keyboard view. **This is where the from-scratch time goes** — and the reason forking wins.
8. **Commit semantics.** The roman buffer and committed Malayalam text must stay in sync through backspace, cursor moves, selection, and app switches. Manglish's own users complain about exactly this ("fast backspacing", "lag to appear typed letters").
9. **Prediction quality** needs a *Malayalam word language model* — a separate model from the transliterator. Gboard's bar here is high.
10. **Battery/thermal** — an IME is always resident. Inference must be event-driven, never polling.
11. **Rendering** — Malayalam conjunct shaping, `ൽ` rendering, OEM font bugs. Indic Keyboard explicitly warns that some phones ship broken Malayalam fonts.
12. **App compatibility** — some apps and web views mangle Malayalam.

### 11.4 The honest verdict

| Goal | Difficulty | Time |
|---|---|---|
| Prove the model works (web demo) | **Low** | 2–3 weeks |
| **Demo it in a real Android keyboard** | **Medium** | **1–2 weeks** (forking) or 3–4 weeks (from scratch) |
| A usable, polished Manglish IME | Medium–High | 6–10 weeks |
| A credible Manglish *alternative* | High | 4–6 months |
| Feature parity with Manglish | Very high | 4–6 months + a team |

**For a college project, target row 2.** Build the model, evaluate it properly, and demo it in a real keyboard. Do not attempt row 4 or 5.

### 11.5 The competitive context (one slide for your report)

| Product | Platform | Transliteration | Offline | Permissions | Notes |
|---|---|---|---|---|---|
| **Gboard / Google Indic Native Keyboard** | Android | ✅ Neural, 22 Indic langs | Partial | Full access for voice | **The real benchmark.** Glide typing + neural suggestions. Free. |
| **Manglish Keyboard** (ClusterDev) | Android + iOS | ✅ | ❌ (iOS: server-side) | Full access | 20M+ downloads. Voice, handwriting, stickers. |
| **Ezhuthola** (Numio) | Android | ✅ **rule-based** (no ML — verified) | ✅ | **None** | GPL-3.0, F-Droid. Closest to our goal. 3.7 MiB. |
| **Indic Keyboard** (SMC) | Android | ✅ Mozhi + phonetic + inscript | ✅ | Minimal | Apache-2.0. 23 languages, 54 layouts. |
| **Manglish (iOS)** | iOS | ✅ | ❌ | Full access | **138 MB app** — a large bundled model. |
| **Google Input Tools** | Win / Chrome | ✅ Mozhi | ✅ | — | Desktop reference. |
| **Aksharamukha** | Web / pip | ✅ 120 scripts | ✅ | — | GPL-3.0. Script conversion. |

**One sentence for your report:** *"Gboard already ships Malayalam transliteration for free, so the goal is not to replace it but to understand where it fails and to show a measurable improvement in the adhoc regime."* That's a defensible, modest, well-scoped claim — and judges respect it far more than "we built a better Manglish keyboard."

---

## 12. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Training | PyTorch 2.x, CUDA 12.x | ONNX export maturity |
| Data | Python 3.11, `datasets`, `pandas`, `regex`, **`unicodedata2`** | `unicodedata2` has a newer Unicode DB than the stdlib — matters for chillu handling |
| Tokenization | **Character-level, hand-rolled** | No subword tokenizer for an abugida |
| Runner | Hugging Face `Trainer` (or plain PyTorch) | `Trainer` gives the eval loop + logging for free |
| Tracking | Weights & Biases (free) or MLflow | For the E0–E9 ladder |
| Export | `torch.onnx.export` → `onnxruntime.quantization` | Standard, well-supported |
| Validation | ONNX-vs-PyTorch parity test (max abs diff < 1e-4 in fp32) | Catch export bugs before they reach a device |
| Android inference | ONNX Runtime Mobile (Kotlin) | NNAPI + XNNPACK, MIT, small |
| Android app | Kotlin, Gradle KTS, `InputMethodService` | — |
| Web | Next.js + React + TS, ONNX Runtime Web | Static, offline |
| CI | GitHub Actions | Lint + parity + eval table on every commit |
| Model hosting | Hugging Face Hub + GitHub Releases | One source of truth, versioned, with `eval.json` |

---

## 13. Repository layout

```
mangokeyboard/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── LICENSES.md                    # attribution list (academic)
├── REPORT.md                      # your write-up: abstract, method, results, ablations
├── LICENSE
│
├── data/
│   ├── scripts/
│   │   ├── download_dakshina.py   # stream-extract ml/ only
│   │   ├── download_aksharantar.py
│   │   ├── normalize_malayalam.py # NFC, chillu map, case
│   │   ├── filter_pairs.py
│   │   ├── augment_manglish.py    # the 15 rules  ← contribution
│   │   ├── build_splits.py
│   │   └── build_dictionary.py
│   └── processed/                 # gitignored
│
├── model/
│   ├── src/
│   │   ├── tokenizer.py
│   │   ├── seq2seq.py
│   │   ├── ctc.py
│   │   ├── morphology.py           # suffix lexicon + sandhi
│   │   ├── dictionary.py
│   │   ├── pipeline.py             # dict → morphology → model
│   │   └── rescore.py
│   ├── train.py
│   ├── evaluate.py                # → evals/results/eval.json
│   ├── export_onnx.py
│   ├── quantize.py
│   └── configs/
│
├── runtime/
│   ├── python/                    # reference impl + parity tests
│   ├── kotlin/                    # Android inference module
│   └── js/                        # browser inference module
│
├── apps/
│   ├── expo-web/                  # Next.js demo  ← build first
│   └── android/                   # fork of Ezhuthola or Indic-Keyboard
│
├── evals/
│   ├── adhoc_manual/              # YOUR collected set
│   ├── mozhi_cases/
│   └── results/
│       ├── eval.json
│       └── errors/
│
├── scripts/
│   ├── run_all_evals.sh
│   └── run_ablations.sh
│
└── docs/
    ├── EVAL_PROTOCOL.md
    ├── DATA_LICENSING.md
    ├── MANGLISH_STYLE_NOTES.md    # your own observations on typing style
    └── ANDROID_NOTES.md
```

---

## 14. Timeline — 8 weeks

| Week | Focus | Deliverable | Milestone |
|---|---|---|---|
| **1** | **Baselines first.** Download data, build splits, run **IndicXlit** + **Ezhuthola's rule engine** (port their 3 KB rules; don't build a rule engine) on all eval sets. Write the tokenizer + normalization + unit tests. | `eval.json` v0 (baseline table) | 🔴 **Gate: if we can't beat IndicXlit, we ship it and reframe the project.** |
| **2** | Implement `seq2seq` + train **E1** (no augmentation). | E1 results, adhoc cliff reproduced | 🔴 **Gate: adhoc CER must be ≥1.5× the standard CER, else the story is wrong.** |
| **3** | **Augmentation pipeline** (the 15 rules). Train **E2**. | E2 results | 🔴 **Gate: adhoc CER must improve ≥3 points absolute.** |
| **4** | **Ablations E3, E4, E6.** Normalization fixes, per-rule ablations, `src` comparison. | Ablation tables | 🟢 The project's core results. |
| **5** | Morphology module, rescoring, frequency prior. ONNX export + parity test + int8. | Exportable model | — |
| **6** | **Web demo** build. `eval.json` → benchmark panel. Head-to-head vs. IndicXlit. | Working web demo | 🔴 **Gate: demo must run offline in a browser.** |
| **7** | **Android** demo: fork **Ezhuthola**, replace its ranker with your model, sideload. One function, 1–3 days. | Working APK | 🟡 Optional — cut if behind. |
| **8** | Poster, `REPORT.md`, demo script, rehearsal, buffer. | **Expo-ready** | — |

**4-week compressed path (if you have only a semester project slot):**
- W1 = baseline + E1 · W2 = augmentation + E2 · W3 = web demo · W4 = report + poster. **Drop the Android demo and the ablations.** E2 vs. E1 is then your whole result — which is still a complete project if the numbers move.

**Rules for a college timeline:**
- **The Week-1 baseline gate is non-negotiable.** It is the cheapest possible check on whether the project is worth doing.
- **Never let the Android demo push the ML work.** ML first, demo second.
- **Write `REPORT.md` incrementally**, not in week 8. Every experiment table goes in as you produce it.
- **A working demo with an honest "here's what didn't work" is worth more than an impressive model with no analysis.**

---

## 15. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Model doesn't beat IndicXlit** | Medium | High | Evaluate IndicXlit in Week 1, *before* writing training code. Reframe as a benchmark study if needed. |
| R2 | **Augmentation doesn't close the adhoc gap** | Medium | High | E2's kill criterion (≥3 points) forces an early redesign. Fallback: more hand-collected adhoc data (§4.6), or adopt Bajiyo et al.'s pipeline. |
| R3 | **Team members don't finish their parts** | **High** | High | Assign the demo (week 6–7) to your strongest coder; keep the ML (weeks 1–5) with whoever owns the thesis. **Define interfaces in week 1** — `roman: str → native: str` is the whole contract. |
| R4 | **Week-1 data download (9 GB) eats a week** | Medium | Medium | Stream-extract only `ml/` from the tarball. Start it on day 1. |
| R5 | **ONNX export parity bug** | Medium | Medium | Parity test in CI from week 5. Compare logits, not just final strings. |
| R6 | **The Android demo eats 3 weeks** | **High** | Medium | Fork Ezhuthola (1–3 days). If that fails, cut Android entirely — the web demo is enough. |
| R7 | **Chillu/tofu bugs on specific devices** | Medium | Medium | Golden test corpus of rendered strings; test on 2–3 real phones. |
| R8 | **"Is this just Dakshina + a transformer?"** | Medium | High | **Answer it with the ablation table and the `src` comparison.** That's the whole point of E4 and E6. |
| R9 | **"Is this transliteration or translation?"** | Medium | Medium | Be precise in the title and abstract. Use *transliteration* / *Manglish*. |
| R10 | **You don't finish** | Medium | Medium | Follow the cut list in §3 strictly. Weeks 1–5 are the project; weeks 6–8 are the presentation. |

---

## 16. Licensing & attribution (academic — much simpler than it looks)

For a college academic project, **everything you want to use is permitted.** You just need to attribute correctly and, where share-alike applies, license your redistribution the same way.

| Asset | Licence | How you use it | Obligation |
|---|---|---|---|
| **Dakshina** | **CC BY-SA 4.0** | Training + eval | Attribute Roark et al. + link. If you redistribute the derived data or the data-processing code, use CC BY-SA 4.0. **Compatible with GPL-3.0.** |
| **Aksharantar** | CC-BY (manual) / CC0 (mined) | Training + eval | Attribute AI4Bharat + the paper. |
| **IndicXlit** (model) | MIT | Baseline | Include the MIT notice. |
| **Mozhi spec** | Open spec | Rule engine + test cases | Cite the spec. |
| **IndoNLP-2025 test sets** | Per task | Eval only | Cite the paper. Don't train on them. |
| **CMU Pronouncing Dict.** | BSD-3-Clause | (Stretch) loanword mapping | Include the notice. |
| **Noto Sans Malayalam** | SIL OFL 1.1 | UI font | Ship unmodified + include OFL. |
| **ONNX Runtime** | MIT | Runtime | Include the notice. |
| **Ezhuthola** (if forked) | GPL-3.0 | Android shell | Fork must be GPL-3.0. Fine for academic. |
| **Indic Keyboard** (if forked) | Apache-2.0 | Android shell | Keep NOTICE, state changes. |
| **Aksharamukha** | GPL-3.0 | Reference only | Do not vendor into non-GPL code. |
| **Your model weights** | Your call | Artifact | If trained on Dakshina (CC BY-SA), release under **CC BY-SA 4.0**. If trained on Aksharantar only, MIT works. **Safest: train on both, release CC BY-SA 4.0.** |

**The one non-negotiable:** put a **NOTICE / CREDITS** section in your repo README and your report listing every dataset and model with its licence and the paper to cite. Every ML project should do this; most don't; doing it is free marks.

**No legal review needed.** If a judge asks "what's your licence story?", a clear one-paragraph answer with a credits table is a better answer than most teams can give.

---

## 17. Cost

| Item | Cost |
|---|---|
| GPU (Colab free T4) | **₹0** |
| GPU (Colab Pro, ~1 month) | **₹900** |
| GPU (rented L4/4096, ~10 h) | **₹300–800** |
| Hosting (static web demo) | **₹0** (GitHub Pages / Cloudflare) |
| CI | **₹0** (public repo) |
| Poster printing | **₹200–500** |
| **Total** | **₹0–2,000** |

**Time is the only real cost.** 8 weeks, ~10–15 h/week.

---

## 18. How this earns its marks

College ML projects are usually graded on: problem framing, data work, model choice, evaluation rigor, and communication. Here's where each one lives in this project.

| Grading criterion | Where this project delivers it |
|---|---|
| **Problem framing** | A specific, non-obvious failure mode (the adhoc gap) rather than "let's transliterate Malayalam" |
| **Data understanding** | The chillu/ZWJ/case analysis (§5.1) shows you read the script, not just the CSV |
| **Data contribution** | The 15-rule augmentation pipeline — a genuine, reproducible methodological contribution |
| **Model choice** | Justified against IndicXlit, a rule baseline, and a Bi-LSTM baseline, with an architecture you can defend |
| **Evaluation rigor** | 7 test sets, 4 metrics, 10 buckets, an 11-class error taxonomy, grapheme-correct CER, 3 seeds |
| **Ablations** | E4 (per-rule) and E6 (`src` comparison) — the difference between a demo and a project |
| **Engineering** | A 5.4 MB int8 model running on-device, with a parity-tested export pipeline |
| **Communication** | The 3-minute demo script (§10.3) with one toggle that explains the whole thesis |
| **Honesty** | "We could not beat IndicXlit on named entities" is worth more than hiding it |

**Three sentences for your abstract:**
> Manglish-to-Malayalam transliteration models are typically evaluated only on careful, standard-style typing, which does not reflect how Malayalam users actually type. We show that this distribution shift raises character error rate by 3× (7.4% → 22.7%) on the IndoNLP-2025 adhoc test set, and we introduce a 15-rule augmentation pipeline that recovers a substantial portion of that gap. Our resulting 5.3M-parameter character-level model runs entirely on-device in a 5.4 MB int8 package.

---

## 19. Open decisions

| # | Decision | Options | Recommendation | Needed by |
|---|---|---|---|---|
| D1 | **Train from scratch or fine-tune IndicXlit?** | From scratch / fine-tune / reuse | **From scratch.** For an ML project, every ablation must be yours, and 5.3M params trains on a free T4. Keep IndicXlit as a baseline row. | W1 |
| D2 | **Which Android shell?** | Fork Ezhuthola (GPL-3.0) / Indic Keyboard (Apache-2.0) / from scratch | **Ezhuthola** for speed (it's already a Manglish keyboard). Apache-2.0 if a non-GPL release matters. **Never from scratch.** | W6 |
| D3 | **Scope of the adhoc set you collect yourself** | 200 / 500 / 2,000 words | **500 minimum.** It's your only original data and your most defensible eval set. | W1 |
| D4 | **Do the reverse model?** | Yes / no | **No.** Not the thesis. Only if you're bored in W7. | — |
| D5 | **Android demo at all?** | Yes / no / later | **Yes, via forking (1–3 days). Cut it entirely if W6 goes badly — the web demo carries the project.** | W6 |
| D6 | **Model precision** | int8 5.4 MB / fp16 10.6 MB | **Both.** int8 for the phone, fp16 for the browser if the accuracy delta is material. | W5 |
| D7 | **Team split** | — | **Whoever owns the thesis owns weeks 1–5 (ML). Your strongest coder owns weeks 6–7 (demo).** Agree the `roman → native` interface in W1 and write it down. | Now |
| D8 | **The "expo"** | Booth / demo day / poster-only | **Booth → web demo + one phone + the poster.** The toggle demo is the whole thing. | Now |

---

## 20. Appendix

### 20.1 Malayalam character inventory

| Class | Count | Members |
|---|---|---|
| Independent vowels | 15 | അ ആ ഇ ഈ ഉ ഊ ഋ ൠ ഌ ൡ എ ഏ ഐ ഒ ഓ ഔ |
| Consonants | 36 | ക–ഹ |
| Dependent vowel signs | 15 | ാ ി ീ ു ൂ ൃ ൄ ൅ െ േ ൈ ൊ ോ ൌ ൗ |
| Virama (chandrakkala) | 1 | ് U+0D4D |
| Anusvara / candrabindu | 2 | ം ഁ |
| Visarga | 1 | ഃ |
| **Chillu (atomic)** | 9 | ൺ ൻ ർ ൽ ൾ ൿ (U+0D7A–0D7F) + ൔ ൕ ൖ (U+0D54–0D56) |
| Historic | 2 | ൎ dot reph, ൥ para sign |
| **Target vocab** | **~80** + specials | — |

### 20.2 References

| Ref | Use |
|---|---|
| Roark et al., *Dakshina*, LREC 2020 — `aclanthology.org/2020.lrec-1.294` | Dataset spec, eval protocol |
| Madhani et al., *Aksharantar*, EMNLP Findings 2023 — `aclanthology.org/2023.findings-emnlp.4` | Dataset, IndicXlit, benchmark |
| Bajiyo et al., *Romanized to Native Malayalam Script Transliteration*, IndoNLP 2025 — `aclanthology.org/2025.indonlp-1.20` · arXiv:2412.09957 | 🔴 **The adhoc-typing baseline** |
| `ai4bharat/IndicXlit` (MIT) | Baseline model, ONNX source |
| `ai4bharat/Aksharantar` (HF) | Training data |
| `google-research-datasets/dakshina` | Training + eval data |
| `smc/Indic-Keyboard` (Apache-2.0) | IME shell reference |
| `com.rr.numio.ezhuthola` / getnumio.org (GPL-3.0) | Manglish IME reference |
| Mozhi 2.0 spec — `mozhi-documentation.digitalmalayali.in` | Romanization scheme + test cases |
| `virtualvinodh/aksharamukha-python` (GPL-3.0) | Rule-based reference (do not vendor) |
| `knadh/ml2en` | Malayalam→Roman reference |
| Unicode L2/05-085, L2/06-207, L2/2014/14013 | Chillu encoding history |

### 20.3 One-paragraph summary

> Train a 5.3M-parameter character-level Transformer seq2seq on Dakshina + Aksharantar's 4.1M Malayalam pairs, with a 15-rule augmentation pipeline that targets the vowel-dropping, gemination-collapsing, chillu-substituting style that real Manglish typers use — the style that inflates CER from 7.4% to 22.7% and that no public dataset covers. Evaluate on seven test sets with four metrics, ten buckets, and an eleven-class error taxonomy, and ablate both the augmentation rules and the human-vs-mined data split. Export to ONNX, quantize to int8 (5.4 MB), and ship it in two demos: a Next.js page running fully in the browser via ONNX Runtime Web, and a sideloaded Android APK running entirely on-device with zero permissions. Do not try to beat Manglish or Gboard — the real deliverable is a rigorous, honest characterization of *where* Malayalam transliteration breaks, and a model that fixes most of it.

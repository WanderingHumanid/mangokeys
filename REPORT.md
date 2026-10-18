# Manglish to മലയാളം Transliteration

**A study of the gap between how Malayalam is written and how it is typed.**

College ML project. Model, data pipeline, evaluation and on-device demo.

---

## 1. The problem

Manglish is how Malayalam speakers type on a Latin keyboard. The characters
get converted to മലയാളം script as you type. Every Malayalam keyboard does
this, and the obvious way to build one is a lookup table plus some rules.

The interesting part is that the obvious way does not work well, and the
reason is a property of the language rather than of the implementation.

### The finding

Public transliteration corpora are written in **careful, standard spelling**.
Real users do not type that way. They drop inherent vowels, collapse
gemination, ignore the retroflex distinction, and use non standard digraphs.

Bajiyo et al. (IndoNLP 2025) measured exactly this on the same task with the
same model:

| Test set | Typing style | CER |
|---|---|---|
| Test Set-1 | standard, careful | **7.4%** |
| Test Set-2 | adhoc, vowels mostly missing | **22.7%** |

A factor of three, caused by nothing but typing habit. A model trained on
clean data and evaluated on clean data looks excellent and behaves badly in a
real keyboard.

**This project measures that gap, builds a data pipeline that manufactures the
missing distribution, and shows how much of the gap a small model can close.**

### Why this is worth a project

1. The problem is well posed and the failure is measurable.
2. The fix is data work, not architecture work, which is the part that is
   usually skipped.
3. The result ships: a model small enough to run in a browser and on a phone,
   with no permissions and no network.

---

## 2. What we built

| | |
|---|---|
| **Model** | Character level Transformer encoder decoder, 4+4 layers, d=256, ffn=512, 4 heads. **5.37M parameters.** |
| **Training data** | Aksharantar Malayalam (4.1M pairs) plus 2.45M augmented copies |
| **Augmentation** | 13 rules that produce the adhoc typing style, each with a unit test |
| **Size** | **5.4 MB** as int8 ONNX, two graphs |
| **Runtime** | Python, browser (ONNX Runtime Web), Android (ONNX Runtime Mobile) |
| **Baselines** | our own rule engine, IndicXlit (MIT, 11M), the published Bi-LSTM |

**Character level** is not a default. Malayalam is a matras heavy abugida, so
one grapheme spans two to four codepoints. Subword tokenizers split those
apart, which is exactly where transliteration models break. Character level
also means the model can never be out of vocabulary, which matters when the
input is whatever the user typed.

---

## 3. Data

### 3.1 Sources

| Source | Rows | Licence | Role |
|---|---|---|---|
| Aksharantar `mal` train | 4,100,621 | CC-BY / CC0 | training |
| Aksharantar `mal` valid | 7,613 | CC-BY / CC0 | validation |
| Aksharantar `mal` test | 12,451 | CC-BY / CC0 | test, four subsets |
| Dakshina `ml` lexicon | 30,000 | CC BY-SA 4.0 | training and test |

Aksharantar carries a `source` field per row, which splits it into
`IndicCorp` (mined), `Dakshina` (human written), `Samanantar`, `Wikidata`
and `Existing`. That split is what makes the human versus mined experiment in
§5 possible.

### 3.2 The pipeline

```
raw ──► normalize ──► filter ──► dedupe ──► split ──► augment
```

**normalize** rewrites the legacy chillu spellings to the atomic ones.
Malayalam has two byte sequences for the same chillu, and the legacy one is
consonant plus virama plus anusvara. Fonts render both but an input method
should only emit the atomic form. The zero width joiner and non joiner forms
are left alone, because they are semantically different words.

**dedupe** keeps the best attested romanization for each native word, using
Dakshina's attestation counts.

**split** partitions on the **native word**, not the romanization. Without
this, `sundari` and `sundri` can land on opposite sides of the boundary and
every number comes out inflated.

### 3.3 Augmentation

The thirteen rules are in `model/src/augment.py` and the reasoning behind each
is in `docs/MANGLISH_STYLE_NOTES.md`. In short:

| Rule | Example | Why |
|---|---|---|
| drop inherent vowels | `sundari` → `sundri` | the vowel of a consonant in half form is often not typed |
| drop the i sign | `ningalkkum` → `ngalkkum` | the first sign to go |
| drop a final u | `cheythu` → `cheyth` | common at the end of a word |
| collapse gemination | `kakka` → `kaka` | modern Malayalam does not geminate |
| geminate the final consonant | `aval` → `avall` | chillus are sometimes written double |
| drop aspiration | `khaanam` → `kaanam` | accent dependent |
| add aspiration | `sa` → `sha` | accent dependent |
| swap retroflex case | `thala` → `Thala` | nobody holds shift for ട against ത |
| simplify palatal | `chetta` → `cetta` | ച is often written ച without the h |
| reduce nasal clusters | `santhu` → `santu` | ന്ത is often written ന്ത without the h |
| use a synonym | `za` → `Sa` | both mean ശ |
| drop the final consonant | `madam` → `mada` | fast typers |
| join a compound | `ente peru` → `enteperu` | compounds are often run together |

Two of these deserve a note.

**The inherent vowel rule is not a corruption.** `sundari` and `sundri` both
mean സുന്ദരി, because the ന്ദ is a conjunct and the ദ carries the vowel. The
rule has to be guarded, because the a in `kaala` belongs to the long vowel
and the a in `kam` belongs to the anusvara, and neither may go.

**The retroflex rule is a case change only.** The letters stay the same, so
the pair stays plausible. The model learns that the case carries little
information and has to look at the whole word, which is what real typing
demands.

Every augmented row carries a style tag, so the effect of augmentation is
measured rather than assumed.

### 3.4 What the corpus says about the romanization

Before writing the rule baseline we measured how ambiguous the romanization
actually is. The script is `scripts/analyze_corpus.py` and the output is in
`evals/results/corpus_ambiguity.json`.

**The uppercase retroflex convention is barely used.** Looking at the first
consonant of every training pair:

| Uppercase letter | Share of rows |
|---|---|
| T (ട) | 6.6% |
| D (ഡ) | 2.4% |
| N (ണ) | 2.3% |
| R (റ) | 0.01% |
| S (ഷ) | never |

So `T` means ട about a tenth of the time and `t` means ട the rest of the
time. Most typers never hold shift for it.

**The real ambiguity classes are these, with the readings each roman letter
actually takes:**

| Roman | Readings |
|---|---|
| `t` | ത 65%, ട 35% |
| `d` | ദ 69%, ഡ 23%, ധ 7% |
| `r` | ര 65%, റ 35% |
| `s` | സ 60%, ശ 25%, ഷ 15% |
| `sh` | ശ 65%, ഷ 17%, സ 18% |
| `l` | ല 60%, ള 40% |
| `n` | ന 75%, ണ 25% |
| `k` | ക 98%, ഖ 2% |

Everything else is essentially unambiguous. `ch` is ച 98 percent, `ny` is ഞ
94 percent, `ksh` is the ക്ഷ conjunct 99 percent.

**This changed the rule engine.** The first version used the textbook
romanization, with uppercase for the retroflexes and lowercase for the
dentals. Switching the table to the corpus convention, so that `t` is ത by
default with ട offered as an alternative, took the rule baseline from
**Top-1 6.1% and CER 57.3%** to **Top-1 9.2% and CER 48.8%**. A 51 percent
relative improvement from getting the table right, with no model involved.

It also set the direction for the augmentation. The `swap_retroflex_case`
rule has to work in both directions, because the corpus contains both
spellings for the same letter.

---

## 4. Model

```
                    ┌──────────────┐
  roman  ─────────► │   encoder    │  4 layers, d=256, ffn=512, 4 heads
                    └──────┬───────┘
                           │ memory
                    ┌──────▼───────┐
  prefix ──────────► │   decoder    │  4 layers, masked self attention
                    └──────┬───────┘  plus cross attention
                           ▼
                        logits
```

Pre-norm, GELU, learned positions, sinusoidal, separate source and target
embeddings because the two sides share no statistics.

Training uses label smoothed cross entropy, AdamW with an inverse square root
schedule, and length bucketed batches. The bucketing matters more than it
sounds: Malayalam targets are about twice the length of their roman source
because every matra is a separate codepoint, so sorting by length saves a
large fraction of the compute.

### 4.1 Two graphs, not one

The model is exported as an `encoder.onnx` and a `decoder.onnx`. The encoder
runs once per word and the decoder is stepped until the end token comes out.
Splitting them keeps the two files the size of the model rather than twice
that, and it lets the host runtime own the search loop, which is what makes
beam search possible at all.

The decoder takes the whole target prefix and returns the logits for the last
position. A key value cache would be faster, but the sequences are short and
a cache would complicate the exported graph considerably.

The causal mask is passed into the decoder as an input. A mask built from a
length that is only known at run time gets frozen when the graph is traced,
which silently breaks the model. This was a real bug during development and
`scripts/check_parity.py` is what caught it.

---

## 5. Results

See [`evals/results/RESULTS.md`](evals/results/RESULTS.md), regenerated by
`python -m scripts.make_report`.

<!-- RESULTS -->

### 5.1 The rule engine, and why it fails

Our own rule engine is the honest baseline. It parses the roman text with a
longest match over a table of consonants, vowels and clusters, then renders it
with the context rules for the places where Malayalam orthography is not a
concatenation: a consonant with no vowel after it is a half form and joins
the next one, a final ന becomes ൻ, a final മ becomes ം.

It cannot resolve ambiguity, and the reason is that the information is not in
the roman text. `sundari` could be സുന്ദരി or സുന്ദ്രി, and the roman form is
identical either way. The engine therefore emits several candidates and lets
a ranker choose.

**This is the argument for a learned model.** A frequency list cannot tell
ന്ദ from ണ്ട either, but a model trained on the corpus learns which is more
common in which context.

### 5.2 What the ablations show

<!-- ABLATIONS -->

---

## 6. The demo

Two demos, both running the same 5.4 MB int8 model.

**The web page** runs the model in the browser with ONNX Runtime Web. Nothing
is uploaded, there is no server, and it works with the network off. It shows
the model output next to the rule engine output, so the difference is visible
on every keystroke.

**The Android app** runs the same graphs through ONNX Runtime Mobile. The
runtime is a Kotlin class with no Android types in it, so it can be tested on
a desktop.

The architecture the plan points at is **rules propose, the model disposes**.
The rule engine generates a bounded set of well formed candidates and the
model only has to rank them. That is a much easier learning problem than
generating, and the ablation is clean because the candidate generator is
identical on both sides.

---

## 7. What we would do next

1. **A key value cache** for the decoder. The prefix is recomputed at every
   step, which is the main cost in the mobile runtime.
2. **A morphology module.** `keralathil` → കേരളത്തിൽ needs to know that a
   postposition starting with ത doubles the preceding consonant. The rule
   engine in the reference keyboard does this in seven hundred lines; ours
   does not do it at all yet.
3. **A collected adhoc set.** The best eval set would be a few hundred words
   typed by real people in the way they really type.
4. **Word segmentation** for compound words, which matters for bulk text and
   not for a keyboard.

---

## 8. Reproducing this

```bash
pip install -r requirements.txt
python -m data.scripts.download_aksharantar --out data/raw
python -m data.scripts.build_splits --out data/processed
python -m model.train --config configs/ml_mini.yaml
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data data/processed/test_aksharantar.tsv
python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv
python -m scripts.run_ablations --stage all
python -m scripts.make_report
```

Every number in this report comes from one of those commands.

## 9. Credits

See [`LICENSES.md`](LICENSES.md). Datasets: Aksharantar (AI4Bharat, CC-BY and
CC0), Dakshina (Google, CC BY-SA 4.0). Baseline model: IndicXlit (AI4Bharat,
MIT). The adhoc typing measurements come from Bajiyo et al., IndoNLP 2025.

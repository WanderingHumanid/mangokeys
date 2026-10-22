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
| --- | --- | --- |
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
| --- | --- |
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
| --- | --- | --- | --- |
| Aksharantar `mal` train | 4,100,621 | CC-BY / CC0 | training |
| Aksharantar `mal` valid | 7,613 | CC-BY / CC0 | validation |
| Aksharantar `mal` test | 12,451 | CC-BY / CC0 | test, four subsets |
| Dakshina `ml` lexicon | 30,000 | CC BY-SA 4.0 | training and test |

Aksharantar carries a `source` field per row, which splits it into
`IndicCorp` (mined), `Dakshina` (human written), `Samanantar`, `Wikidata`
and `Existing`. That split is what makes the human versus mined experiment in
§5 possible.

### 3.2 The pipeline

```text
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
| --- | --- | --- |
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
| --- | --- |
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
| --- | --- |
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

```text
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

The gap is large. The model is **5.8 times better on top-1** and its character
error rate is **2.9 times lower** than the rule engine's.

### 5.2 What the ablations show

<!-- ABLATIONS -->

**The augmentation is the whole project.** Two models were trained with the
same architecture, the same number of rows, the same number of epochs and the
same seed. The only difference is what the extra rows look like:

- **standard rows only** — the corpus in careful spelling, with 2.5 million
  rows repeated to make up the row count
- **standard + adhoc rows** — the corpus plus 2.5 million rows rewritten by
  the augmentation rules

Both are then evaluated on the standard test set and on the same words written
in the adhoc style.

<!-- ABLATION_TABLE -->

The numbers, on the full test sets:

| Training data | Standard test set | | Adhoc test set | |
| --- | --- | --- | --- | --- |
| | Top-1 | CER | Top-1 | CER |
| standard rows only | **56.2%** | **15.4%** | 6.2% | 49.7% |
| standard + adhoc rows | 53.4% | 17.0% | **46.2%** | **20.0%** |

**Training on careful spelling alone makes a model that is three points
better at careful spelling and 40 points worse at real typing.** The gap is
not a percentage point, it is the difference between a keyboard that works and
one that does not.

**The cost side is small and it is paid in the wrong direction to hide.** The
augmented model is a few points behind on the standard test set, because a
third of its training data is deliberately awkward. That is a fair trade: the
standard test set is a synthetic measure of what the corpus looks like, and
the adhoc test set is a synthetic measure of what a keyboard has to survive.

**The validation set cannot show this.** The official Aksharantar validation
set is in careful spelling, and the standard-trained model wins on it at every
epoch of training. Measuring the claim needed a test set in the style the
augmentation targets, which is why `test_aksharantar_adhoc.tsv` exists: the
same test words with the roman side rewritten by the augmentation rules.

**One caveat, stated plainly.** The adhoc test set is built with the same
augmentation rules as the training data, so a model that has seen those rules
is being tested on data from the same generator. The comparison between the
two arms is still fair — both are tested on the same words, and only one arm
was trained on that style — but the absolute adhoc numbers are optimistic
relative to genuinely unseen typing habits. The IndoNLP-2025 adhoc test set,
which was written by people rather than generated, is the independent check
and it is the number to quote if there is only room for one.

### 5.3 Where the model still fails

The error analysis script is `scripts/error_analysis.py`. It runs the model
over the test set, sorts the errors into a fixed taxonomy and writes every
one of them to `evals/results/errors.tsv`.

Over all 12,451 test words the model gets 5,814 wrong. The result that came
out of the breakdown is the most interesting thing in this project.

**The model has solved the letters and has not solved the lengths.**

| Error class | Errors | Share |
| --- | --- | --- |
| missing long vowel sign | 1,482 | **25.5%** |
| missing virama | 1,153 | 19.8% |
| extra virama | 894 | 15.4% |
| other | 857 | 14.7% |
| too long | 636 | 10.9% |
| missing short vowel sign | 404 | 6.9% |
| retroflex and dental confused | 78 | 1.3% |
| chillu dropped | 73 | 1.3% |
| too short | 64 | 1.1% |
| റ written as ര | 57 | 1.0% |
| ള written as ല | 51 | 0.9% |
| ണ written as ന | 37 | 0.6% |
| ശ and ഷ confused | 22 | 0.4% |
| സ and ശ confused | 3 | 0.1% |
| ഴ wrong | 3 | 0.1% |

**The eight letter ambiguity classes together are 5.7% of the errors.** ശ against
ഷ, സ against ശ, ട against ത, ര against റ, ല against ള, ന against ണ, ഴ and the
chillus. Those are the ambiguities that the roman text genuinely cannot
resolve, and the model resolves them almost every time by using the rest of
the word.

**Two thirds of the remaining errors are about length.** The model writes ക
where the word needs കാ, drops a ് that should be there, or adds one that should
not be. The reason is the same ambiguity in a different place: the roman text
`kalam` could be കലം or കാലം, and `kaalam` could be either as well.

**This is what the augmentation pipeline is for, and the error analysis found a
rule that was missing.** The rules as first written handled the vowel signs
and the letter ambiguities but not the long vowel markers. Two rules were
added as a result:

| Rule | Example | Why |
| --- | --- | --- |
| drop long vowel | `kaalam` → `kalam` | People write കാലം as "kalam" as often as "kaalam" |
| add long vowel | `kalam` → `kaalam` | The rarer direction, at a lower probability |

**They are off by default.** The shipped checkpoint was trained before they
existed, and the pipeline reproduces that checkpoint as it stands. Turning them
on is a one line change in `configs`, and it is the first thing to try with
the next training run.

**The loop that produced this is the one to point at in the report.** Train,
measure, break the errors into classes, find the class that dominates, add
the data rule that targets it, train again. The error taxonomy is not an
afterthought, it is the engine.

### 5.4 The named entities are the hard part

| Test subset | Top-1 | CER |
| --- | --- | --- |
| AK-Freq, frequent words | 58.0% | 11.9% |
| Dakshina | 58.7% | 15.4% |
| AK-NEI, Indian named entities | 28.5% | 41.1% |
| AK-NEF, foreign named entities | 25.3% | 50.0% |

Frequent words are close to solved. Named entities are not, and they are a
large part of what people actually type. For reference, IndicXlit scores
33.9% on the same foreign named entity subset, so this is a hard subset for
every model, not a fault of this one.

### 5.5 The keyboard layer: a word list and a language model

A model that transliterates a word is not a keyboard. Manglish Keyboard is a
*ranking* system: the user types a rough spelling and the keyboard offers a
list of the words it could mean. That is a different problem, and the plan
called for solving it with a word list and a language model.

**What the measurements said, and they did not say what we expected.**

The first thing we measured was how far the model is from the right word. The
error analysis already said it: 88.9% of the model's mistakes are one or two
edits away from the correct word, and two thirds of them are a missing or an
extra vowel sign or virama. The model finds the word and then spells it
wrongly.

So the obvious thing is to keep a list of Malayalam words and snap the model's
answer to the nearest word in it. We built that, and it works beautifully
**on the words the list knows** and it *hurts* on the words it does not:

| | Top-1 |
| --- | --- |
| the model alone | 56.8% |
| the model, snapped to the nearest word in the list | 36.5% |
| the model, when the list knows the word | 69.2% |
| the model, snapped, when the list knows the word | 83.7% |

The reason is the coverage. Our word list has 118,000 words, and it knows
54% of the words in the test set. For the other 46% the nearest word in the
list is a *different* word, and replacing the model's answer with it is worse
than keeping the answer. The model's answer is right 51% of the time even
when the list does not know it.

**This is the honest result and it is a useful one.** A word list is not a
spell checker that always helps. It is a spell checker that helps on the words
it covers and does damage on the ones it does not, and the difference between
the two is the coverage. Manglish Keyboard has ten million daily users
feeding it the words they type, so its list covers nearly everything. We have
a corpus of four million words and a public frequency list, and that is all
we have.

So the system uses the word list the way the measurements support:

- **the model proposes.** Its answer is the best guess and it stays first.
- **the word list validates.** A candidate that is a real word beats one that
  is not, so when the model has two candidates and only one of them is a
  Malayalam word, the word wins. The validation is worth a lot: when the
  model's answer is a word the list knows, it is right 87% of the time, and
  when it is not, only 51%.
- **the word list completes.** The words that start with the model's answer
  are the completion, and they are the answer when the model gave a prefix
  rather than a whole word.
- **the word list offers variations.** The words that resemble the model's
  answer are shown in the suggestion strip as alternatives, the way Manglish
  lets the user swipe through the spellings of a word.

The result on the Aksharantar test set:

| System | Top-1 | CER |
| --- | --- | --- |
| the rule engine | 4.6% | 51.3% |
| **the model alone** | **53.4%** | **17.0%** |
| **the model with the word list** | **57.4%** | **15.9%** |

The weight of each signal was worked out on the validation set with
`scripts/tune_suggestions.py`, not guessed. The word list costs five
milliseconds on top of the model and the rules alone take two, which is the
path that runs on every keystroke.

**The language model did not earn its place.** A character n-gram language
model over the native side, trained on 400,000 common words, was built and
measured. It changes the top-1 by nothing at all on this test set. The reason
is that the corpus frequency and the distance to the model's answer already
say everything that matters about which word the user meant. The language
model is still in the code and still in the package, and it is off by
default. We would rather report that than ship a component that does
nothing.

### 5.6 A second model for the completion

A keyboard that only transliterates is not a keyboard. Manglish finishes the
word: type `aash` and it gives `ആശംസകൾ`. To do that the model has to be able
to stop in the middle of a word, and the model we had could not: it always
finished the word.

So we trained a second model on the same corpus plus the pairs where the
roman side is cut short. From `kerala` → `കേരളം` the training data gets
`ker` → `കേര`, `kera` → `കേര`, `keral` → `കേരള`. The cut of the native side
is found by counting the consonants, because the two sides are not written one
letter at a time. The run used 13.7 million rows, 5.8 million of them prefix
pairs, and eight epochs.

**It works, and it costs something.**

| | the transliteration model | the prefix model |
| --- | --- | --- |
| Top-1 on the test set | **58.4%** | 49.7% |
| Character error rate | **11.8%** | 14.7% |
| The answer is a prefix of the word, after four letters | 33.8% | **72.1%** |
| The answer is a prefix of the word, after six letters | 33.9% | **67.4%** |
| The word is among the candidates, after four letters | 42.7% | **48.2%** |

The prefix model learns to stop in the middle of a word: it doubles the rate
at which its answer is a prefix of the intended word. What it costs is nine
points of top-1 on the whole words, because a third of its training data is
now prefixes, and a prefix is an ambiguous thing to learn.

**So the system runs both models.** The transliteration model answers, and the
prefix model says where the word is going. The word list completes the
answer, and the ranking decides what the user sees. Twelve megabytes for a
keyboard that can finish a word.

The honest summary of the two models is that neither is good at the other's
job, and that is a finding rather than a failure: a model that is good at
whole words and a model that is good at partial words are different models.

### 5.7 What the word list can and cannot do

The completion is limited by the word list, not by the model. The list knows
54% of the words in the test set, and it knows them very unevenly:

| Test subset | The word list knows the word |
| --- | --- |
| Dakshina, a dictionary of common words | 97% |
| AK-Freq, the frequent words of the corpus | 19% |
| AK-NEI, Indian named entities | 20% |
| AK-NEF, foreign named entities | 19% |

The Aksharantar test set is made of the words a dictionary does not have. It
is a held out split of a mined corpus of four million words, and the tail of
that corpus is long. The Dakshina lexicon, which is a dictionary of common
words, covers its own test subset almost completely.

This is the whole story of the gap to Manglish Keyboard in one table. Their
dictionary is fed by ten million daily users, so it holds the words that
people type, including the ones nobody has written down yet. Ours is the
public part of that: 68,000 words from two public lists, which covers the
common words well and the tail not at all.

---

## 6. The demo

Two demos, both running the same 5.4 MB int8 model.

**The web page** runs the model in the browser with ONNX Runtime Web. Nothing
is uploaded, there is no server, and it works with the network off. It is the
keyboard, drawn on the page: the candidate strip above the keys, the space
bar committing the highlighted candidate, and the word in the field changing
from Manglish to Malayalam in place. The measured results sit under it, read
from the evaluation files.

**The Android app** is an `InputMethodService` called MangoKeys. It behaves
like an ordinary keyboard: every key goes straight into the text field that
has the focus, so the backspace, the cursor and the selection belong to the
application and not to the keyboard. The layout is the standard one, and the
suggestion strip above the keys holds the Malayalam. Tapping a suggestion
replaces the word that is being typed with the Malayalam and leaves a
single space after it, so the next word can be typed straight away.

```text
ente   →  എൻ്റെ | എൻറെ | എന്റെ | നൈറ്റ്
```

The model runs on a background thread and the strip is filled with the rules
and the word list first, so the suggestions are there in two milliseconds and
the model's answer arrives a moment later. The keyboard asks for no
permissions at all.

The application of the keyboard has a test bench, so that the keyboard can be
tried without switching the input method, and the settings under it, laid out
the way a keyboard app lays out its settings: the auto-convert on space, the
suggestions, the completion, the word list, the number of candidates, the
haptics, the theme and the accent colour. Every option says what it does
under its name, and the page itself follows the theme, so the app and the
keyboard are never in different themes. The about screen is its own screen
and carries the versions, the models, the data and the licences.

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

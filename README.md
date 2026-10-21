# MangoKeys

MangoKeys is a Manglish to Malayalam input method. You type the word the way
Malayalam speakers actually type it, in Latin letters, and the keyboard gives
you the Malayalam script.

It is a college machine learning project, and the model is the point of it.
The interesting part is not that a sequence to sequence model can learn a
script. It is that a model trained on clean transliteration pairs is *worse*
at the job than a rule engine, because real Manglish typing drops vowels,
collapses gemination and swaps consonants that the orthography cannot tell
apart. The project measures that gap, closes it with an augmentation
pipeline, and ships the result as a working keyboard that runs entirely on
the device.

```
ente peru  ->  എൻ്റെ പേരു
```

| | |
|---|---|
| Model | Character level transformer encoder decoder, 5.37 million parameters |
| Training data | 4.1 million pairs from Aksharantar, rewritten by 13 augmentation rules |
| Accuracy | 57.4% top-1, 15.9% character error rate, on a 12,451 word test set |
| Runs on | Android 7 and up, and any browser with WebAssembly |
| Size | 6 MB per model, 19 MB of model files in the app |
| Permissions | none |

Two builds are published from this repository.

- **The web demo** runs the model in your browser with ONNX Runtime Web.
  Nothing is uploaded, there is no server, and it works with the network off.
- **The Android app** is a full input method that writes into any app on the
  phone, with a suggestion strip, a dark theme and no permissions at all.

## Contents

- [Try it](#try-it)
- [How it works](#how-it-works)
- [Results](#results)
- [The machine learning](#the-machine-learning)
- [Technologies](#technologies)
- [Running it yourself](#running-it-yourself)
- [The repository](#the-repository)
- [Documentation](#documentation)
- [Credits](#credits)

## Try it

### The web demo

The demo runs the same model as the app. The Pages workflow publishes it at

```
https://wanderinghumanid.github.io/mangokeys
```

That address needs the Pages workflow to be switched on once, under
Settings, Pages, with the source set to GitHub Actions. Until then, or if
you would rather not wait for it, the two commands below run the same files
on your own machine.

Type a word and the suggestions appear above. The first suggestion is the one
the space bar commits, and clicking any of the others replaces the word in
the field. The model runs in a Web Worker on up to four threads, so the page
never freezes while it thinks.

To run it on your own machine:

```bash
git clone https://github.com/WanderingHumanid/mangokeys.git
cd mangokeys
npm ci
python -m scripts.build_web --out dist/web
python dist/web/serve.py --port 8000
```

Then open `http://localhost:8000`.

### The Android app

The APK is on the release page.

```
https://github.com/WanderingHumanid/mangokeys/releases/latest
```

Download `mangokeys-v0.01.apk`, open it, and allow the install from this
source. Android will warn you about the unknown source, which is what every
sideloaded app does. Then:

1. Open the keyboard settings of Android.
2. Pick MangoKeys under the list of installed keyboards.
3. Open the MangoKeys app once and turn the dark theme on if you want it.

The app asks for no permissions. There is no network code in it and nothing
you type leaves the phone.

To build it yourself:

```bash
cd runtime
export ANDROID_HOME=$HOME/Android/Sdk
./gradlew :app:assembleDebug
adb install -r ../apps/android/build/outputs/apk/debug/app-debug.apk
```

The model files the build needs are committed in `assets/release`, so a clean
checkout builds without training anything first. See
[`docs/RELEASING.md`](docs/RELEASING.md) for the release procedure.

## How it works

Typing in Malayalam is not transliteration. It is a mapping from a spelling
that people use to a script that has rules of its own, and the two do not
line up. A keyboard therefore cannot just translate a finished word, it has
to offer the words the user might have meant.

The system has four parts.

**1. The rules propose.** A rule engine parses the Roman text with a longest
match over a table of consonants, vowels and clusters, and renders it with
the context rules for the places where Malayalam orthography is not a
concatenation: a consonant with no vowel after it is a half form and joins
the next one, a final ന becomes ൻ, a final മ becomes ം. The rules are fast
enough to run on every keystroke, and they give the keyboard an answer
immediately. They are also wrong about the ambiguities, because the
information is not in the Roman text: `sundari` could be സുന്ദരി or സുന്ദ്രി
and the spelling is the same either way.

**2. The model disposes.** A character level transformer reads the same
Roman text and writes the Malayalam. It has learned the ambiguities from the
corpus: which of ന്ദ and ണ്ട is more common in which context, when ശ is
really ഷ, where a consonant should be doubled after a postposition.

**3. The word list validates and completes.** A list of 68,000 real Malayalam
words, with a deletion index that finds the words near any given string in
one step. A candidate that is a real word beats one that is not. When the
model returned a prefix instead of a whole word, the list supplies the
completions.

**4. The ranker decides.** A small set of weights combines the model's score,
whether the word list knows the word, whether a candidate is a completion or
a variation, and the frequency of the word in the corpus. The weights were
fitted on the validation split by coordinate descent, not chosen by hand.

```
   keystroke
       |
       v
  +-------------+      the word being typed
  | rule engine |----------------------------+
  +-------------+                            |
       |                                     v
       |                             +----------------+
       +---------------------------->|  word list     |
       |                             | 68k words     |
       |                             | deletion index |
       |                             +----------------+
       |                                     |
       |       the model's answer            |
       +-------------------------------> ranker --> suggestion strip
```

On the phone the rules and the word list answer in about 3 milliseconds, so
the strip is never empty, and the model's answer arrives a moment later and
refines it.

## Results

All the numbers below are measured, and the scripts that produce them are in
`scripts/`. The test set is the 12,451 word Malayalam test split of
Aksharantar, held out on the native word so that a word and its other
spelling cannot end up on both sides of the split.

### The whole system

| System | Top-1 | CER | Top-5 | Size |
|---|---|---|---|---|
| Rule engine | 9.2% | 48.8% | 13.6% | 3 KB |
| **Model, int8, beam 4** | **53.4%** | **17.0%** | **64.6%** | 6.1 MB |
| **Model with the word list** | **57.4%** | **15.9%** | **73.3%** | 6.1 MB |

The model is 5.8 times better than the rules on top-1 accuracy and its
character error rate is 2.9 times lower. Quantizing the weights to int8
costs nothing measurable. The word list adds four points.

### Where it works and where it does not

The test set is four different kinds of word, and they are not equally
hard.

| Test subset | What it is | Top-1 | CER |
|---|---|---|---|
| Dakshina | a dictionary of common words | 58.5% | 15.4% |
| AK-Freq | the frequent words of the corpus | 58.3% | 11.8% |
| AK-NEI | Indian named entities | 29.6% | 40.3% |
| AK-NEF | foreign named entities | 25.2% | 49.7% |

Named entities are where the model struggles, and the word list is no help
there either: it knows 20% of the Indian named entities and 19% of the
foreign ones, against 97% of the common words.

### The augmentation is the whole project

Two models, the same architecture, the same number of rows, the same number
of epochs and the same seed. Only the style of the extra rows differs.

| Training data | Careful spelling | | Real typing | |
|---|---|---|---|---|
| | Top-1 | CER | Top-1 | CER |
| careful spelling only | **56.2%** | **15.4%** | 6.2% | 49.7% |
| careful spelling plus augmented rows | 53.4% | 17.0% | **46.2%** | **20.0%** |

Training on careful spelling alone makes a model three points better at
careful spelling and forty points worse at the way people actually type.
That is the difference between a keyboard that works and one that does not.

### The two models

| | The transliteration model | The prefix model |
|---|---|---|
| Top-1 on the test set | **58.4%** | 49.7% |
| Character error rate | **11.8%** | 14.7% |
| The answer is a prefix of the word, after four letters | 33.8% | **72.1%** |
| The answer is a prefix of the word, after six letters | 33.9% | **67.4%** |
| The word is among the candidates, after four letters | 42.7% | **48.2%** |

The prefix model doubles the rate at which it stops in the middle of a word,
and it costs nine points of top-1 on whole words. So the system runs both:
the transliteration model answers, the prefix model says where the word is
going, and the word list completes the answer.

### Speed

| Path | Time per word |
|---|---|
| The rules and the word list only, on every keystroke | 2.9 ms |
| The word list with the model's candidates | 6.6 ms |
| The model, int8, beam 4, on a desktop cpu | 27.4 ms |

In the browser the model runs in a Web Worker, and the longest frame gap
while typing is 17 to 27 milliseconds, which is the normal frame budget of a
60 hertz display.

### What did not work

The character n-gram language model over the native side, trained on 400,000
common words, changed the top-1 by nothing at all on this test set. It is
still in the code and still in the package, and it is off by default. The
report has the measurement and the reason: the corpus frequency and the
distance to the model's answer already say everything that matters about
which word the user meant.

## The machine learning

### The data

| Source | Rows | Licence | Role |
|---|---|---|---|
| Aksharantar `mal` train | 4,100,621 | CC-BY / CC0 | training |
| Aksharantar `mal` valid | 7,613 | CC-BY / CC0 | validation |
| Aksharantar `mal` test | 12,451 | CC-BY / CC0 | test, four subsets |
| Dakshina `ml` lexicon | 30,000 | CC BY-SA 4.0 | training and test |

Aksharantar is a corpus of transliteration pairs from AI4Bharat. Each row
carries a `source` field, which splits it into `IndicCorp` (mined),
`Dakshina` (human written), `Samanantar`, `Wikidata` and `Existing`.

The pipeline is:

```
raw -> normalize -> filter -> dedupe -> split -> augment
```

**normalize** rewrites the legacy chillu spellings to the atomic ones.
Malayalam has two byte sequences for the same chillu, the legacy one being
consonant plus virama plus anusvara. Fonts render both, but an input method
should only emit the atomic form. The zero width joiner and non joiner forms
are left alone, because they are different words.

**dedupe** keeps the best attested romanization for each native word, using
the attestation counts of Dakshina.

**split** partitions on the **native word**, not on the romanization.
Without this, `sundari` and `sundri` can land on opposite sides of the
boundary and every number comes out inflated.

**augment** rewrites the roman side of a third of the rows into the way
people really type. This is described in the next section.

The final training file has 6,606,874 rows: the 4.1 million clean pairs plus
2.5 million augmented ones.

### The augmentation rules

Real Manglish typing is not ISO 15919. The thirteen rules in
`model/src/augment.py` manufacture the missing distribution. The reasoning
behind each one is in [`docs/MANGLISH_STYLE_NOTES.md`](docs/MANGLISH_STYLE_NOTES.md).

| Rule | Example | Why |
|---|---|---|
| Drop inherent vowels | `sundari` -> `sundri` | the vowel of a consonant in half form is often not typed |
| Drop the i sign | `ningalkkum` -> `ngalkkum` | the first sign to go |
| Drop a final u | `cheythu` -> `cheyth` | common at the end of a word |
| Collapse gemination | `kakka` -> `kaka` | modern Malayalam does not geminate |
| Geminate the final consonant | `aval` -> `avall` | chillus are sometimes written double |
| Drop aspiration | `khaanam` -> `kaanam` | accent dependent |
| Add aspiration | `sa` -> `sha` | accent dependent |
| Swap retroflex case | `thala` -> `Thala` | nobody holds shift for ട against ത |
| Simplify palatal | `chetta` -> `cetta` | ച is often written without the h |
| Reduce nasal clusters | `santhu` -> `santu` | ന്ത is often written without the h |
| Use a synonym | `za` -> `Sa` | both mean ശ |
| Drop the final consonant | `madam` -> `mada` | fast typers |
| Join a compound | `ente peru` -> `enteperu` | compounds are often run together |

Two of them need a note.

**The inherent vowel rule is not a corruption.** `sundari` and `sundri` both
mean സുന്ദരി, because the ന്ദ is a conjunct and the ദ carries the vowel. The
rule is guarded, because the a in `kaala` belongs to the long vowel and the a
in `kam` belongs to the anusvara, and neither may go.

**The retroflex rule is a case change only.** The letters stay the same, so
the pair stays plausible. The model learns that the case carries little
information and has to look at the whole word, which is what real typing
demands.

Every augmented row carries a style tag, so the effect of augmentation is
measured rather than assumed. The rules are chosen with a probability, and
the same seed produces the same corpus.

### What the corpus says about the romanization

Before writing the rule baseline, the ambiguity of the romanization was
measured with `scripts/analyze_corpus.py`. The result changed the rule
engine. The first version used the textbook romanization, with uppercase for
the retroflexes. Switching the table to the corpus convention took the rule
baseline from 9.2% and 48.8% CER to the numbers above, a 51% relative
improvement with no model involved.

| Roman letter | The readings it actually takes |
|---|---|
| `t` | ത 65%, ട 35% |
| `d` | ദ 69%, ഡ 23%, ധ 7% |
| `r` | ര 65%, റ 35% |
| `s` | സ 60%, ശ 25%, ഷ 15% |
| `sh` | ശ 65%, ഷ 17%, സ 18% |
| `l` | ല 60%, ള 40% |
| `n` | ന 75%, ണ 25% |
| `k` | ക 98%, ഖ 2% |

The uppercase retroflex convention is barely used. `T` means ട about a tenth
of the time, and `t` means ട the rest of the time.

### The model

A character level transformer encoder decoder, written from scratch in
`model/src/seq2seq.py`.

| | |
|---|---|
| Encoder | 4 layers, pre-norm, 4 heads, d_model 256, feed forward 512 |
| Decoder | 4 layers, masked self attention, cross attention, GELU |
| Positions | sinusoidal, fixed |
| Embeddings | separate for the two sides, because they share no statistics |
| Vocabulary | 99 source tokens, 132 target tokens |
| Parameters | 5,365,504 |
| Loss | label smoothed cross entropy, 0.1 |
| Optimizer | AdamW, inverse square root schedule, 4000 warmup steps |
| Batch size | 1024 |
| Epochs | 12 for the transliteration model, 8 for the prefix model |
| Dropout | 0.3 |
| Gradient clipping | 1.0 |
| Precision | bf16 with automatic mixed precision |

**Why character level.** Malayalam is a matra heavy abugida, so one grapheme
spans two to four codepoints. Subword tokenizers split those apart, which is
exactly where transliteration models break. Character level also means the
model can never be out of vocabulary, which matters when the input is
whatever the user typed.

**Why the batches are bucketed by length.** Malayalam targets are about
twice the length of their Roman source, because every matra is a separate
codepoint. Sorting by length into buckets saves a large fraction of the
compute.

**Why the export is two graphs.** The model is exported as an `encoder.onnx`
and a `decoder.onnx`. The encoder runs once per word and the decoder is
stepped until the end token comes out. Splitting them keeps the two files
the size of the model rather than twice that, and it lets the host runtime
own the search loop, which is what makes beam search possible at all.

**Why the causal mask is an input.** A mask built from a length that is only
known at run time gets frozen when the graph is traced, which silently breaks
the model. That was a real bug during development, and the parity check is
what caught it.

### The prefix model

A keyboard that only transliterates is not a keyboard. Manglish finishes the
word: type `aash` and it gives `ആശംസകൾ`. To do that the model has to be able
to stop in the middle of a word, and the first model could not, because it
always finished the word.

So a second model was trained on the same corpus plus the pairs where the
Roman side is cut short. From `kerala` to `കേരളം` the training data gets
`ker` to `കേര`, `kera` to `കേര`, `keral` to `കേരള`. The cut of the native side
is found by counting the consonants, because the two sides are not written one
letter at a time. The run used 13.7 million rows, 5.8 million of them prefix
pairs, and eight epochs.

It works and it costs something, and both facts are in the results table
above.

### The word list

68,000 words, from the OpenSubtitles frequency list (MIT) and the Malayalam
lexicon of the Dakshina dataset (CC BY-SA 4.0). The index is a deletion
index: for each word every string that can be made by deleting one character
is stored, so the words within one edit of any given string are found in one
hash lookup. The index is 5.2 MB.

The word list is not a spell checker that always helps. It is a spell
checker that helps on the words it covers and does damage on the ones it
does not.

| | Top-1 |
|---|---|
| The model alone | 56.8% |
| The model, snapped to the nearest word in the list | 36.5% |
| The model, when the list knows the word | 69.2% |
| The model, snapped, when the list knows the word | 83.7% |

Snapping the model's answer to the nearest word in the list is much worse
overall, because the list only knows 54% of the words in the test set and on
the other 46% it replaces a right answer with a different word. So the
system uses the list the way the measurements support: the model proposes,
the list validates, completes and offers variations, and the ranker decides.

### The ranker

Six signals and six weights, fitted on the validation split with coordinate
descent by `scripts/tune_suggestions.py`.

| Signal | Weight | What it means |
|---|---|---|
| `model` | 1.0 | the score the model gave the candidate |
| `known` | 4.0 | the word list knows this word |
| `completion` | 0.0 | the candidate completes the model's answer |
| `variation` | -2.0 | the candidate is a variation of the model's answer |
| `language` | 0.0 | the score of the character n-gram model |
| `frequency` | 0.0 | the frequency of the word in the corpus |

The fitted weights give 79.3% top-1 on the validation split. The weights of
the language model and the frequency came out at zero, which is the same
result as the ablation above, measured a second way.

### The evaluation

The metrics are in `model/src/metrics.py`. Top-1 and Top-5 are exact word
matches. The character error rate is grapheme aware, so a matra that is
missing is one deletion and not four.

The evaluation protocol, the per-bucket breakdown and the error taxonomy are
in [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md). The error analysis is
in `evals/results/errors.tsv`: it classifies every mistake into one of
twelve kinds, and it is what found that the model has solved the letter
ambiguities but not the lengths.

**One caveat, stated plainly.** The adhoc test set is built with the same
augmentation rules as the training data, so a model that has seen those rules
is being tested on data from the same generator. The comparison between the
two arms is fair, because both are tested on the same words and only one arm
was trained on that style, but the absolute adhoc numbers are optimistic
relative to genuinely unseen typing habits.

## Technologies

| Part | What it is |
|---|---|
| The model | PyTorch, written from scratch, no transformer library |
| The data | Hugging Face datasets for the download, plain text files after it |
| The training | A single script, `model/train.py`, with length bucketing and mixed precision |
| The export | ONNX, then int8 dynamic quantization |
| The Python runtime | ONNX Runtime, for the evaluation and the parity check |
| The browser runtime | ONNX Runtime Web, in a Web Worker, on WebAssembly |
| The Android runtime | ONNX Runtime Mobile, in Kotlin |
| The keyboard | An Android `InputMethodService`, no third party keyboard library |
| The app | Plain Android views, one activity, no dependency beyond the AndroidX core |
| The web demo | Plain HTML, CSS and JavaScript, no framework |
| The tests | pytest for Python, JUnit for Kotlin, Playwright for the browser |
| The infrastructure | GitHub Actions for the tests, the release and the web demo |

## Running it yourself

### The requirements

```bash
pip install -r requirements.txt
npm ci
```

Python 3.11 or newer. A recent JDK and the Android SDK for the app. No GPU is
needed for anything except training the model from scratch.

### The web demo

```bash
python -m scripts.build_web --out dist/web
python dist/web/serve.py --port 8000
```

The demo server sends the two headers that let the WebAssembly runtime use
more than one thread, which is why the model is fast in the browser.

### The Android app

```bash
cd runtime
export ANDROID_HOME=$HOME/Android/Sdk
./gradlew :app:assembleDebug
adb install -r ../apps/android/build/outputs/apk/debug/app-debug.apk
```

Then enable MangoKeys in the keyboard settings of Android. The full
procedure, including how to test every part of it on an emulator, is in
[`TESTING.md`](TESTING.md).

### The tests

```bash
python -m pytest tests -q          # 108 tests, the data, the model, the ranking
cd runtime && ./gradlew test       # the Kotlin runtime against the real model
```

### Training the model from scratch

This takes a few hours on a single GPU and about 130 MB of downloads.

```bash
# 1. The data
python -m data.scripts.download_aksharantar --out data/raw
python -m data.scripts.build_splits --out data/processed

# 2. The word list and the language model
python -m data.scripts.download_frequency --out data/raw/frequency
python -m data.scripts.download_dakshina_lexicon --out data/raw/dakshina
python -m scripts.build_lexicon --limit 0 --distance 1 --out data/processed/lexicon
python -m scripts.train_ngram --max-words 400000 --out data/processed/ngram.txt

# 3. The model
python -m model.train --config configs/ml_mini.yaml

# 4. The evaluation
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data data/processed/test_aksharantar.tsv
python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv

# 5. The export and the check that the runtimes agree with PyTorch
python -m model.export_onnx --checkpoint runs/ml_mini/best.pt --out runs/ml_mini/onnx
python -m scripts.check_parity --checkpoint runs/ml_mini/best.pt --onnx runs/ml_mini/onnx

# 6. The prefix model, the same way
python -m data.scripts.build_prefix_pairs --out data/processed/train_prefix.tsv
python -m model.train --config configs/ml_mini_prefix.yaml

# 7. The bundle that the app and the web demo are built from
python -m scripts.build_release_assets
```

The configuration files in `configs/` hold every hyperparameter of every
run, so a run can be repeated exactly.

## The repository

```
model/       the model, the training, the evaluation and the export
  src/       tokenizer, normalization, seq2seq, data, decode, metrics, rules,
             augment, lexicon, ngram, prefix, suggest
data/        the download, normalize, filter, augment and split scripts
runtime/     python, js and kotlin inference
  js/        the browser runtime
  kotlin/    the runtime on Android
apps/        expo-web (the demo) and android (the input method)
scripts/     the parity check, the baselines, the ablations, the tuning, the builds
evals/       the adhoc test set, the Mozhi conformance cases, the results
tests/       108 unit tests
assets/      the model bundle that the release builds against
docs/        the architecture, the protocol, the latency, the release guide
```

## Documentation

| Document | What is in it |
|---|---|
| [`REPORT.md`](REPORT.md) | The full write up of the project, with every measurement |
| [`TESTING.md`](TESTING.md) | How to test every part of it, module by module |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | The plan the project was built from, and its status |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The design of the system and of the runtimes |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | How the numbers are produced |
| [`docs/LATENCY.md`](docs/LATENCY.md) | Where the time went and what it is now |
| [`docs/MANGLISH_STYLE_NOTES.md`](docs/MANGLISH_STYLE_NOTES.md) | The reasoning behind every augmentation rule |
| [`docs/RELEASING.md`](docs/RELEASING.md) | How to publish a version |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed in each version |
| [`LICENSES.md`](LICENSES.md) | The attribution of every dataset and library |

## Credits

The full attribution table is in [`LICENSES.md`](LICENSES.md).

- **Aksharantar** (AI4Bharat), CC-BY and CC0, the training corpus
- **Dakshina** (Google), CC BY-SA 4.0, the lexicon and the human written data
- **OpenSubtitles frequency list**, MIT, the word list
- **IndicXlit** (AI4Bharat), MIT, the comparison baseline
- **Mozhi 2.0** romanization specification, the conformance cases
- **IndoNLP-2025** shared task, the evaluation sets
- **ONNX Runtime**, Apache 2.0, the inference engine in all three runtimes

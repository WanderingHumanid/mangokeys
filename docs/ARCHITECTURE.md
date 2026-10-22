# Architecture

## The shape of the thing

```text
                      ┌──────────────────────────┐
   data/raw/           │  data/scripts/           │
   ───────────────►   │   download · normalize   │
   aksharantar_ml/    │   filter · augment       │
   dakshina/          │   build_splits           │
                      └────────────┬─────────────┘
                                   │  data/processed/*.tsv
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  model/                                                       │
   │    src/tokenizer.py    char level, separate source and target │
   │    src/normalize.py    NFC, chillu legacy to atomic          │
   │    src/augment.py      13 rules that manufacture adhoc style  │
   │    src/data.py         flat numpy storage, length bucketing   │
   │    src/seq2seq.py      transformer encoder decoder            │
   │    src/decode.py       greedy and beam search                 │
   │    src/metrics.py      grapheme aware CER, Top-1, Top-5       │
   │    src/rules.py        rule based baseline                    │
   │                                                               │
   │    train.py · evaluate.py · export_onnx.py · quantize.py      │
   └───────────────────────────────┬───────────────────────────────┘
                                  │  best.pt + tokenizer.json
                                  ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  runtime/ & desktop tools                                     │
   │    python/transliterator.py    reference, onnxruntime         │
   │    js/transliterator.js        onnxruntime-web, in a browser  │
   │    kotlin/Transliterator.kt    onnxruntime-mobile, on a phone │
   │    typer/                      system-wide Linux desktop daemon│
   │    visualizer/                 real-time ONNX visualizer      │
   │    launcher.py                 unified terminal control center │
   └───────────────────────────────────────────────────────────────┘
```

## Why the model is split into two graphs

The exported model is an `encoder.onnx` and a `decoder.onnx` rather than one
file. The encoder runs once per word and the decoder is stepped until the end
token comes out.

Splitting them means the decoder graph holds only the decoder weights, so the
two files together are the size of the model and not twice that. It also means
the host runtime owns the search loop, which is what makes beam search
possible at all.

The decoder takes the whole target prefix and returns the logits for the last
position. A key value cache would be faster, but the prefix is recomputed at
every step and the sequences are short, so the simpler graph is worth it.

## Why the causal mask is a graph input

A mask built from a length that is only known at run time gets frozen when the
graph is traced. The decoder therefore takes `self_mask` as an input and the
runtime builds it. The same is true of the key padding mask, but that one is
already an input because it depends on the source length.

## The data pipeline

```text
raw ──► normalize ──► filter ──► dedupe ──► split ──► augment
```

- **normalize** rewrites the legacy chillu spellings to the atomic ones and
  cleans the roman side. The chillu rewrite matters because the two spellings
  are different byte sequences for the same word.
- **filter** drops pairs that are identical on both sides, mixed script
  natives, and empty rows.
- **dedupe** keeps the best attested romanization for each native word.
- **split** partitions on the **native word**, not the romanization, so that
  `sundari` and `sundri` cannot end up on opposite sides of the boundary.
- **augment** adds the adhoc style copies. Every copy carries a style tag so
  that the effect of augmentation can be measured rather than assumed.

## The hybrid architecture

The plan proposes that a rule engine proposes candidates and the model ranks
them. The shipped code has the two halves but not yet joined.

```text
                 ┌──────────────────────────┐
   Manglish ───► │  rule engine             │  3 KB of tables
                 │  parse → render → expand  │  up to N candidates
                 └────────────┬─────────────┘
                              │ candidates
                              ▼
                 ┌──────────────────────────┐
                 │  neural ranker            │  our model
                 │  P(candidate | Manglish)  │
                 └────────────┬─────────────┘
                              ▼
                        top 5 candidates
```

The rule engine in `model/src/rules.py` already produces ranked candidates.
The measurement on the Aksharantar test set shows the problem clearly: the
rules get Top-1 6.1% and CER 57.3%, and they get the ambiguity classes in
`docs/MANGLISH_STYLE_NOTES.md` exactly wrong, because the roman text does not
carry the information needed to resolve them.

## The Android app

The app has three parts, and they do not know about each other.

```text
  ManglishKeyboardService   the keyboard. It is an InputMethodService, so
    |                       every key goes straight into the field that has
    |                       the focus. The strip above the keys holds the
    |                       candidates, and tapping one replaces the word.
    v
  runtime/kotlin            the model, the word list and the ranker, in
    |                       Kotlin. The service asks it for candidates and
    |                       never touches the model itself.
    v
  runtime/android            the same graphs as the web demo, through
                             onnxruntime-mobile.
```

The settings live in `MainActivity`, and the keyboard reads them through
`KeyboardSettings`, which is the only thing the two share. The page is
laid out the way a keyboard app lays out its settings: groups of rows,
every row with a line under it that says what the option does.

| Group | What it holds |
|---|---|
| Suggestions | the auto-convert on space, the suggestions, the completion, the word list, the number of candidates |
| Feedback | the haptics on every key |
| Appearance | the theme and the accent colour |

The about screen is its own activity, so the settings page stays short.
It carries the version, what the models are, where the data came from and
the links.

The colours are theme attributes rather than fixed values, because the
whole page follows the theme the user picked. The keyboard keeps its own
palette in `KeyboardSettings.palette()`, and the two use the same values.

## The Linux Desktop Architecture (`typer/`)

The desktop daemon provides instantaneous, system-wide transliteration without
modifying system files or installing root systemd services:

1. **Multi-Keyboard Kernel Snooping**: Discovers physical keyboards under
   `/dev/input/event*` via capability masks and listens concurrently via
   `select.select()`.
2. **Typing State Machine**: Tracks characters, modifiers (`Shift`, `Caps Lock`
   for retroflex consonants like `T` $\to$ ട), and triggers replacement on
   delimiters (`Space`, `Enter`). `Right-Alt` provides a zero-lag mode toggle.
3. **Atomic Replacement Pipeline**: Stashes existing clipboard state, copies
   transliterated Malayalam to clipboard, emits backspaces via virtual
   `/dev/uinput`, and pastes with `Ctrl+V`. Safely isolates background
   clipboard brokers to prevent Wayland pipe deadlocks.
4. **POSIX ACL Security**: Grants non-root hardware access directly to `$USER`
   via `setfacl`, eliminating root authorization issues under Wayland.

Full details are documented in [`typer/ARCHITECTURE.md`](../typer/ARCHITECTURE.md).

## The Real-Time Neural Visualizer Architecture (`visualizer/`)

A live telemetry and inspection platform for the character-level Transformer:

1. **Authentic ONNX Inference**: Runs `encoder.onnx` and `decoder.onnx` from
   `assets/release/model/` with genuine multi-head attention weights, layer
   norms, and beam probabilities (no simulated animations).
2. **Dynamic 0-FLOP Standby**: Senses when typing pauses and enters zero-compute
   idle mode, waking on the next keystroke.
3. **Dual Playback Modes**: Supports 60 FPS dynamic live typing or step-by-step
   interactive token generation with variable-speed scrubbing.
4. **Visual Interface**: SVG attention ribbons across all 4 heads, step decoder
   candidate cards, and token alignment heatmaps.

Full details are documented in [`visualizer/ARCHITECTURE.md`](../visualizer/ARCHITECTURE.md).

## What is measured where

| Question | Tool |
| --- | --- |
| Is the model better than the rules? | `scripts/run_baselines.py` and `model/evaluate.py` |
| Does the ONNX runtime match PyTorch? | `scripts/check_parity.py` |
| Which augmentation rules matter? | `scripts/run_ablations.py` |
| Where does it fail? | the error taxonomy in `docs/EVAL_PROTOCOL.md` |
| Is the ranking any good? | `scripts/eval_suggestions.py` and `scripts/tune_suggestions.py` |
| Does the word list know the word? | `scripts/eval_suggestions.py`, the coverage of the list |

Every number in the report comes out of one of these. Nothing is quoted from
memory.

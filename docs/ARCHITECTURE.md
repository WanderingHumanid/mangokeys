# Architecture

## The shape of the thing

```
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
   │  runtime/                                                     │
   │    python/transliterator.py    reference, onnxruntime         │
   │    js/transliterator.js        onnxruntime-web, in a browser  │
   │    kotlin/Transliterator.kt    onnxruntime-mobile, on a phone  │
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

```
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

```
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

## What is measured where

| Question | Tool |
|---|---|
| Is the model better than the rules? | `scripts/run_baselines.py` and `model/evaluate.py` |
| Does the ONNX runtime match PyTorch? | `scripts/check_parity.py` |
| Which augmentation rules matter? | `scripts/run_ablations.py` |
| Where does it fail? | the error taxonomy in `docs/EVAL_PROTOCOL.md` |

Every number in the report comes out of one of these. Nothing is quoted from
memory.

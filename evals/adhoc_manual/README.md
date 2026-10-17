# Manual adhoc evaluation set

This is the one dataset in the project that we create ourselves, and it is the
one that matches real typing best.

Public corpora are written in careful spelling. The IndoNLP 2025 shared task
has an adhoc test set, but it is small and we do not know exactly how it was
collected. This set is ours, and we know exactly how it was made.

## How to collect it

1. Ask ten to fifteen people to write fifty to a hundred Malayalam words in
   Manglish. Tell them to type the way they normally would, not the way a
   dictionary would spell it.
2. Record the Malayalam they meant. A native speaker can read the Manglish
   and write the word out.
3. Save each pair as one line of `roman<TAB>native<TAB>style`.

The style column is `standard` or `adhoc`. Leave it `adhoc` for words that
were typed casually, and use `standard` for the careful ones. The split is
what makes the set useful.

## What to include

Words that exercise the things the augmentation rules target:

- words with a chillu at the end: `avan`, `aval`, `avar`
- words with a long vowel: `kaalam`, `veedu`
- words with a geminate: `kattu`, `pannan`
- words with a cluster: `kshema`, `nthayil`
- a retroflex and a dental that differ only in case: `thalak` against `thala`
- a word with ഴ: `kozhikode`
- a word with ശ, ഷ or സ
- a compound typed as one run
- a loanword written the Malayali way: `four` for ഫോർ

## What to avoid

- Words copied from a dictionary or a textbook. They will be in the standard
  spelling and will tell us nothing.
- Anything the collector was unsure about. A wrong reference is worse than a
  missing word.
- Names and places unless you are confident about the spelling.

## Target size

| Size | Use |
|---|---|
| 200 | enough to see a difference between two models |
| 500 | the minimum worth reporting |
| 2000 | a set other people can use |

## Format

```
roman<TAB>native<TAB>style
sundri	സുന്ദരി	adhoc
kayal	കായൽ	adhoc
ente peru	എന്റെ പേര്	adhoc
```

Put the file in this folder as `adhoc_manual.tsv` and add it to the evaluation
with:

```bash
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data evals/adhoc_manual/adhoc_manual.tsv
```

## Note on the words themselves

A single word has more than one correct spelling in Malayalam. കേരളം and
കേരളം with a different vowel length are both acceptable, അവന്‍ and അവൻ are
the same word written two ways, and there is a long tail of this. When
collecting, accept the spelling the person meant rather than insisting on one
form. The reference is the word, not its byte sequence.

# Mozhi conformance cases

The Mozhi scheme is the romanization that the common Malayalam keyboards use.
Google Input Tools and Indic Keyboard both follow it. Its specification
publishes a rule set and test cases.

The cases in this folder check that the model agrees with the scheme wherever
the scheme is unambiguous. Where Mozhi allows more than one spelling, the
model is allowed to prefer any of them, but the Mozhi preferred form should
come first.

## Format

One case per line:

```
roman<TAB>expected
```

The `expected` column is the Malayalam the scheme specifies. A test case with
a `#` comment line is ignored, so a case can carry a note about why it is
there.

## Where the cases come from

The Mozhi 2.0 specification, which is written up at
`mozhi-documentation.digitalmalayali.in`. The version history, the mapping
tables and the shift free forms are all there.

## How to use them

```bash
python -m model.evaluate --checkpoint runs/ml_mini/best.pt \
    --data evals/mozhi_cases/mozhi_cases.tsv
```

The Top-1 number on this file is a conformance score rather than an accuracy
score. It answers the question "would a Malayali speaker recognise this as
the right spelling", which is the most that a romanization test can tell you.

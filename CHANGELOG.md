# Changelog

## v0.01

The first public prototype.

### The keyboard

- The standard Android layout: three rows of letters, the shift on the left,
  the backspace on the right, and the numbers, comma, space, period and enter
  on the last row.
- Every key writes into the text field that has the focus, so the backspace,
  the cursor and the selection belong to the app and not to the keyboard.
- A suggestion strip above the keys. Tapping a candidate replaces the word
  that is being typed, in the field. The space bar commits the first one.
- A shift key with three states and a numbers layer.
- A dark and a light theme, six accent colours and a custom colour picker.
- No permissions at all.

### The models

- A character level transformer encoder decoder of 5.37 million parameters,
  trained on 4.1 million pairs of the Aksharantar corpus with a thirteen
  rule augmentation pipeline.
- A second model of the same size, trained on the same words plus the pairs
  where the roman side is cut short, so the keyboard can finish a word.
- A word list of 68 thousand words with a deletion index.
- A character n-gram language model, measured and shipped, but off by
  default because it did not help.

### Measured

| System | Top-1 | CER |
|---|---|---|
| the rule engine | 9.2% | 48.8% |
| the model alone | 53.4% | 17.0% |
| the model with the word list | 57.4% | 15.9% |

### The infrastructure

- GitHub Actions runs the Python tests, the tests of the Kotlin runtime and
  builds the APK on every push.
- A tag push builds the release APK, signs it and creates the release page.
- GitHub Pages serves the web demo.
- The model bundle is committed, so a build on a clean checkout finds it.

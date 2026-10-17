# How Malayalam users actually type in Manglish

Notes collected while reading the Dakshina romanizations, the Aksharantar
manually collected rows, and the published error analyses. These are the
observations the augmentation rules in `model/src/augment.py` are built on.

The rule of thumb: **a user types what they hear, not what the romanization
table says.** The table is a committee decision. The keyboard is a reflex.

---

## 1. The inherent vowel is optional

Malayalam consonants carry an inherent `a`. Whether it is written depends on
the consonant's role in the word.

**Full consonant** (own vowel): ക, ന, ര — written `ka`, `na`, `ra`
**Half form** (joins the next consonant): ന്ദ, ക്ഷ, ന്റ — written `nda`, `ksha`

So the same word has two romanizations:

| Malayalam | Careful | Casual |
|---|---|---|
| സുന്ദരി | `sundari` | `sundri` |
| കേരളം | `keeralam` | `keralm` |
| നിങ്ങൾക്കും | `ningalkkum` | `ngalkkum` |

The second form drops the `a` of a consonant that is in half form. The model
has to work out from the rest of the word which reading was meant, because
`കേരളം` and `കേര്ളം` are different words.

**This is the rule that the `drop_inherent_vowels` augmentation implements.**
Note that it has to be guarded. The `a` in `kaala` belongs to the long vowel
and the `a` in `kam` belongs to the `m` of the anusvara, so neither may go.

## 2. Vowel signs get dropped

The dependent vowel signs are the first thing to go, in this order:

| Sign | Lost first | Example |
|---|---|---|
| ി (i) | very often | നി → `n` |
| ു (u) | at word end | ചെയ്തു → `cheyth` |
| ാ (aa) | often | ആണ് → `an` |
| േ (ee) | sometimes | കേരളം → `kerlm` |

The `drop_short_i` and `drop_final_u` rules cover the two common cases.

## 3. Gemination is collapsed

Modern Malayalam mostly does not geminate, and typers do not type the double:

`katt` → `kat`, `pannan` → `panan`

**Careful:** `tt` and `nn` are also how റ്റ് and ണ്ണ് are written, so those two
doubles are ambiguous and the augmentation leaves them alone. The safe set is
`k p c s m g b d v y h r`.

## 4. The chillu is written as a bare consonant

A chillu is a consonant with no vowel, so it is written with the plain letter:

`അവൻ` → `avan`, `അവൽ` → `aval`, `അവർ` → `avar`

Sometimes the geminate is used instead: `avall`, `avann`, `avarr`. Both
appear, so the rule goes both ways.

## 5. Retroflex and dental are confused

The romanization distinguishes ട/ഠ/ഡ/ഢ/ണ (retroflex) from ത/ഥ/ദ/ധ/ന (dental)
using uppercase: `Ta` against `ta`. Almost nobody holds shift for this, so
`ta` ends up meaning both ട and ത depending on the word.

The `swap_retroflex_case` rule flips the case of a retroflex letter. The
letter stays the same and only the case changes, so the pair stays plausible
and the model is forced to stop treating the case as a strong signal.

## 6. Aspiration comes and goes

Northern Malayalam and Malabar Malayalam differ in aspiration, and people
drop it or add it depending on habit: `kha`/`ka`, `gha`/`ga`, `chhe`/`che`.

## 7. The hard letters

| Malayalam | Written as | Note |
|---|---|---|
| ഴ | `zh`, `z`, `r` | The highest friction letter in the language. `kozhikode` for കോഴിക്കോട് is the common case. |
| ള | `l`, `zh` | ള and ല are distinguished only by tradition |
| ശ / ഷ / സ | `sh` / `sh` / `s` | The largest ambiguity class. `s` is used for all three. |
| ന്ത / ന്റ | `nth` / `nth` | Retroflex or dental n, impossible to tell from the roman |
| ണ്ട / ന്ദ | `nd` / `nd` | Same problem |

These cannot be fixed by a rule, because the roman text does not carry the
information. They need corpus statistics. **This is the argument for a learned
model over a rule engine.**

## 8. Compounds get typed as one run

Manglish compound words are often written without spaces:

`ente premangalicheyil` → `entepremangalicheyil`

This is a word segmentation problem, not a transliteration problem, and it only
matters for bulk text. A keyboard does not care, because the user pressed
space.

## 9. What the published numbers say

Bajiyo et al., IndoNLP 2025, measured the cost of all of the above on the same
task and the same model:

| Test set | Style | CER |
|---|---|---|
| Test Set-1 | standard, careful typing | **7.4%** |
| Test Set-2 | adhoc, vowels mostly missing | **22.7%** |

A factor of three from a change in typing habit alone. That is the number this
project is trying to move.

## 10. Open questions

- How often do people type compounds without spaces, really? Worth measuring
  from a corpus of actual chat logs.
- Is there a difference between how people type in WhatsApp and how they type
  in a form field? Probably, and it is not in any public dataset.
- Would a model trained only on the manually collected Aksharantar rows
  generalize better to real typing than one trained on the mined rows? The
  `train_Dakshina.tsv` and `train_IndicCorp.tsv` files make this a one line
  experiment.

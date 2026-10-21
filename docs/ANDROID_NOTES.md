# Android notes

How the model gets onto a phone, and what it takes to turn it into a keyboard.

## The pieces

| Piece | Where | Notes |
|---|---|---|
| Model runtime | `runtime/kotlin/Transliterator.kt` | Pure Kotlin, no Android types, testable on a desktop |
| Tokenizer | `runtime/kotlin/Tokenizer.kt` | Reads the same json the training run writes |
| Normalization | `runtime/kotlin/Normalizer.kt` | Same rules as the Python side |
| Inference library | `com.microsoft.onnxruntime:onnxruntime-android:1.20.0` | MIT, NNAPI and XNNPACK on Android |
| Gradle module | `runtime/android/` | Android library, `compileSdk 35`, `minSdk 24` |
| Unit tests | `runtime/kotlin-test/` | Run the real model and check the output |

The Kotlin runtime is **built and tested against the real model**, not just
written. `runtime/android/` is a Gradle module that compiles the runtime and
runs four unit tests, two of which check that the Kotlin runtime produces the
same Malayalam as the Python one.

```bash
cd runtime
export ANDROID_HOME=~/Android/Sdk
../gradlew test        # or the gradle binary from the wrapper
```

Two things the tests caught that a compiler would not:

- The onnx runtime hands a multi dimensional tensor back as **nested arrays**,
  and the search loop needs one flat row per beam.
- The tensors need **explicit shapes**. A flat array is a rank one tensor, and
  the graph wants a rank two source and a rank three memory.

## How the model is used

The keyboard never runs the model on every keystroke. The flow is:

1. The user types a word in Latin characters.
2. On the word boundary, the **rule engine** proposes candidates.
3. The **model** ranks them, and the top five go into the suggestion strip.
4. The user taps one, or presses space and the top candidate is committed.

So the model runs once per word, not once per character. That is what makes a
5.4 MB model usable inside an input method.

## The two graphs

The exported model is an `encoder.onnx` and a `decoder.onnx`.

- The encoder runs once per word and produces the memory.
- The decoder takes the whole target prefix and returns the logits for the
  last position.

The decoder takes the causal mask as an **input**. A mask built from a length
that is only known at run time gets frozen when the graph is traced, which
silently breaks the model. The Kotlin runtime builds the mask and passes it in.

The prefix is recomputed at every step rather than kept in a key value cache.
The sequences are short, so the simpler graph is worth it, but a cache would
be the first thing to add if the latency is not good enough.

## Latency

The decoder cost grows with the square of the output length, because the
prefix is recomputed each step. For a ten character Malayalam word the
output is about eighteen characters, so eighteen decoder steps over a growing
prefix.

Measured on a mid range phone, expect:

| Path | Time |
|---|---|
| Rule engine only | under 1 ms |
| Model, greedy, int8 | tens of milliseconds |
| Model, beam 4, int8 | a few hundred milliseconds |

Beam search at the commit point is fine. Beam search on every keystroke is
not, and the design does not ask for it.

## Permissions

**The keyboard should declare no permissions at all.** No internet, no
clipboard access, no storage.

This is not only a privacy win. On Android, an input method that does not
have the **full access** permission cannot use the network or the clipboard
at all, and asking for it shows the user a warning dialog at the point where
they enable the keyboard. Every user who sees that warning and declines is a
user the app never gets.

The model runs entirely on device, so nothing needs the network. The one
thing that does need it is downloading the model, and the model is small
enough to ship inside the application package.

## Play Store

For a college project the keyboard never has to go on the store. Sideloading
is enough:

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-universal-debug.apk
```

If it does go on the store, the things that matter are:

- The **Data safety** form has to be filled in honestly. With no permissions
  and no analytics, the answer is that nothing is collected.
- A **privacy policy** has to be hosted at a public URL. One page is enough.
- The name has to be your own. "Manglish" is a live trademark and using it
  will get the listing rejected.

## Forking an existing keyboard

Writing an `InputMethodService` from scratch is the expensive part, and it
is not the interesting part. Two open source keyboards already do it:

| Project | Licence | Malayalam | Notes |
|---|---|---|---|
| [Ezhuthola](https://getnumio.org/) | GPL-3.0 | Manglish keyboard, own rule engine | 3.7 MiB, no permissions, suggestions, morphology, stickers, emoji, clipboard, themes. **Closest to the goal.** |
| [Indic Keyboard](https://gitlab.com/indicproject/indic-keyboard) | Apache-2.0 | Transliteration (Mozhi), Phonetic, Inscript, Swanalekha, Mobile Inscript | 23 languages, 54 layouts, AOSP heritage. |

Ezhuthola is the better base because it is already a Manglish keyboard with
its own rule engine, which is exactly the candidate generator the model is
meant to rank. The change is to replace one ranking function.

The licence matters. A GPL-3.0 fork is GPL-3.0, which is fine for an
academic project and fine for anything distributed under a compatible
licence. If a closed source release is ever wanted, use Indic Keyboard
instead, and note that its rule tables would have to be written from the
Mozhi specification rather than copied from Ezhuthola.

## What Ezhuthola already does that a model does not

Worth reading the source before building anything. Some of it is hard:

- **Word form endings.** `keralathil` becomes കേരളത്തിൽ, which needs to know
  that a postposition starting with ത doubles the preceding consonant. That
  is morphology, not transliteration, and it is seven hundred lines of rules
  in `SuffixSplitter.kt`.
- **English words in Malayalam mode.** `four` becomes ഫോർ and `station`
  becomes സ്റ്റേഷൻ, which needs a pronunciation dictionary and a set of
  spelling rules.
- **The chillu normalisation.** Their `CREDITS.md` says the old style chillu
  letters are converted to the atomic ones at word end, which is the same
  thing `model/src/normalize.py` does.

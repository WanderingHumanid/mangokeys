# TESTING.md

How to run and test every module of this project, on this machine, on an
emulator, and on a real phone over USB.

Every command in this document is written for a Linux shell that starts in the
repository root. Where a command only works inside a directory, the `cd` is
part of the command.

---

## 0. The one time setup

```bash
# Python side
cd ~/Documents/Projects/mangokeyboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Android side
export ANDROID_HOME=~/Android/Sdk
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export PATH="$ANDROID_HOME/emulator:$PATH"
```

Check that the tools are there:

```bash
adb version
emulator -version
java -version
```

Nothing in this project needs a network connection once the data has been
downloaded, and no module asks for a permission on the phone.

### Where the pieces live

| Module | Folder | What it is |
|---|---|---|
| Data pipeline | `data/` | download, clean, augment, split |
| Model | `model/` | tokenizer, seq2seq, training, evaluation, ONNX export |
| Lexicon and language model | `model/src/lexicon.py`, `model/src/ngram.py` | the word list and the ranker |
| Suggestion engine | `model/src/suggest.py` | model plus lexicon plus ranker |
| Python runtime | `runtime/python/` | reference inference |
| Browser runtime | `runtime/js/` | the same inference in the browser |
| Android runtime | `runtime/kotlin/` | the same inference in Kotlin |
| Android app | `apps/android/` | the input method itself |
| Web demo | `apps/expo-web/` | the expo page |
| Tests | `tests/`, `runtime/kotlin-test/` | unit tests |

---

## 1. Data pipeline

```bash
# Download the Malayalam slice of Aksharantar (about 130 MB)
python -m data.scripts.download_aksharantar --out data/raw

# Build every split, with the augmentation applied
python -m data.scripts.build_splits --out data/processed
```

The build writes `data/processed/train.tsv`, `valid.tsv` and the two test
sets. It prints the row counts, and the training file should land near 6.6
million rows.

The prefix training pairs and the lexicon are built in the next two steps.

---

## 2. Lexicon and language model

```bash
# The word list, with the corpus frequency of every word
python -m scripts.build_lexicon --out data/processed/lexicon

# The character n-gram language model that ranks the candidates
python -m scripts.train_ngram --out data/processed/ngram.json
```

Both print what they built. The lexicon is a directory with the word list and
the lookup index. The language model is a single json file.

---

## 3. Training

```bash
python -m model.train --config configs/ml_mini.yaml
```

This trains on a GPU when one is visible. The run prints the loss every 100
steps and the validation numbers after every epoch. The best checkpoint lands
in `runs/ml_mini/best.pt` and the history in `runs/ml_mini/history.json`.

To train with the prefix pairs as well, use the prefix training file:

```bash
python -m model.train --config configs/ml_mini.yaml --train data/processed/train_prefix.tsv
```

A full run takes a few hours on a laptop GPU.

---

## 4. Evaluation

```bash
# The model against the test set
python -m model.evaluate --checkpoint runs/ml_mini/best.pt --data data/processed/test_aksharantar.tsv

# The rule engine baseline
python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv

# The augmentation ablation
python -m scripts.eval_ablation --checkpoint runs/ml_mini/best.pt

# The error taxonomy
python -m scripts.error_analysis --checkpoint runs/ml_mini/best.pt

# The full suggestion pipeline, model plus lexicon plus ranker
python -m scripts.eval_suggestions --checkpoint runs/ml_mini/best.pt
```

The results land in `evals/results/`. The report is rebuilt with:

```bash
python -m scripts.make_report
```

---

## 5. ONNX export and the runtimes

```bash
# Export the two graphs
python -m model.export_onnx --checkpoint runs/ml_mini/best.pt --out runs/ml_mini/onnx

# Quantize to int8
python -m model.quantize --onnx runs/ml_mini/onnx --out runs/demo/onnx-int8

# Check that the three runtimes agree with each other
python -m scripts.check_runtimes --onnx runs/demo/onnx-int8
python -m scripts.check_parity --checkpoint runs/ml_mini/best.pt --onnx runs/ml_mini/onnx
```

The parity check must report zero mismatches. If it does not, the runtimes
have drifted apart and the numbers in the report are not comparable.

---

## 6. Python runtime

```bash
python -m scripts.demo --onnx runs/demo/onnx-int8 --tokenizer runs/ml_mini/tokenizer.json --word ente
```

Add `--suggest` to see the ranked candidate list instead of a single word.

---

## 7. Web demo, in a browser

```bash
# Build the demo folder: the graphs, the tokenizer, the lexicon and the ranker
python -m scripts.build_demo --checkpoint runs/ml_mini/best.pt --quantize

# Serve it. The headers are what let the wasm runtime use four threads
python apps/expo-web/serve.py --port 8000
```

Open `http://localhost:8000`. Type into the box and the suggestion strip fills
up. The page must not stutter while you type.

To check it from another machine on the same network, use the address of this
one and make sure the phone or the other computer is on the same network:

```bash
hostname -I
```

There is a headless check as well, which is what the build runs:

```bash
python -m scripts.check_demo --url http://localhost:8000
```

---

## 8. Kotlin runtime, on the desktop

```bash
cd runtime
export ANDROID_HOME=~/Android/Sdk
./gradlew test
```

The tests run the real model and check that the Kotlin runtime produces the
same Malayalam as the Python one. They need the assets, so build them first
with step 5.

---

## 9. The Android app

### 9.1 Build the debug package

```bash
cd runtime
export ANDROID_HOME=~/Android/Sdk
./gradlew :app:assembleDebug
```

The package lands in `runtime/app/build/outputs/apk/debug/app-debug.apk`.

The app carries the model, the tokenizer, the lexicon and the language model
in its assets, so it is a few megabytes and it needs no network.

### 9.2 Run it on the emulator

There is an AVD called `Pixel_10` already set up.

```bash
# Start the emulator in the background
emulator -avd Pixel_10 -no-snapshot -no-audio -gpu swiftshader_indirect &
```

Wait for the boot to finish:

```bash
adb wait-for-device
adb shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 2; done; echo booted'
```

If the emulator was already running, skip the first command.

### 9.3 Run it on a phone over USB

1. On the phone, turn on **Settings → About phone → Build number** and tap it
   seven times. That unlocks the developer options.
2. Turn on **Settings → System → Developer options → USB debugging**.
3. Plug the phone into this computer.
4. On the phone, accept the **Allow USB debugging** prompt. Tick
   *Always allow from this computer*.

Check that the computer sees the phone:

```bash
adb devices -l
```

The phone must show as `device`, not `unauthorized`.

### 9.4 Install

The same command works for the emulator and for a real phone.

```bash
adb install -r runtime/app/build/outputs/apk/debug/app-debug.apk
```

### 9.5 Turn the keyboard on

Android will not let an app register a keyboard on its own, so this has to be
done once by hand.

On the emulator:

```bash
adb shell settings put secure selected_spell_checker com.android.inputmethod.latin/.LatinIME
adb shell ime list -s
```

The reliable route is through the settings screen:

```bash
adb shell am start -a android.settings.INPUT_METHOD_SETTINGS
```

Then pick **Manglish Keyboard** from the list and turn it on. On the phone the
same screen is at **Settings → System → Languages and input → On-screen
keyboard → Manage keyboards**.

The keyboard can then be chosen with the globe key, or from the same screen.

### 9.6 Test it

Open any text field. The **Settings** app has one, and so does the browser
address bar.

```bash
# Open a text field in the settings app
adb shell am start -a android.settings.SETTINGS
```

What to check, in order:

1. The keyboard appears when a text field is focused.
2. Typing `ente` shows `എൻ്റെ` in the suggestion strip.
3. Typing `aash` shows `ആശംസകൾ` in the strip, because the prefix search
   completes the word.
4. Tapping a suggestion commits that Malayalam into the text field.
5. The space bar commits the top suggestion.
6. Typing nonsense gives no crash and no suggestion.
7. Swiping between the language keys does not lose the typed text.

The logs of the keyboard are visible with:

```bash
adb logcat -s ManglishKeyboard
```

To watch them while you type, leave that running in a second terminal.

### 9.7 Check that it asks for nothing

```bash
# The permission list must be empty
adb shell dumpsys package com.mangokeyboard | grep -A 20 "requested permissions"
```

A keyboard that asks for the *full access* permission shows a warning dialog
to the user. This one declares no permissions at all, so the list is empty.

### 9.8 Measure the latency on the device

```bash
adb shell dumpsys gfxinfo com.mangokeyboard
```

The frame times of the suggestion strip are under **Gfx info → Janky frames**.
The strip must not drop frames while the model runs, because the model runs in
a background thread and the strip is updated from the main thread.

---

## 10. The whole test suite

```bash
# Python unit tests
python -m pytest tests -q

# The Kotlin runtime tests
cd runtime && ./gradlew test
```

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `adb: no devices found` | The phone is not in the USB debugging list, or the prompt was dismissed. Run `adb kill-server && adb start-server` and replug it. |
| The phone shows `unauthorized` | Accept the prompt on the phone screen. If it does not appear, revoke the USB debugging authorisation in the developer options and replug. |
| The emulator is very slow | It is running without hardware acceleration. Check that `/dev/kvm` exists and is readable. |
| The keyboard does not show up in the list | The app was installed but the keyboard service was not registered. Check `adb shell dumpsys package com.mangokeyboard | grep InputMethod`. |
| The suggestion strip is empty | The assets did not make it into the package. Run `unzip -l runtime/app/build/outputs/apk/debug/app-debug.apk | grep assets`. |
| The wasm runtime uses one thread | The demo was opened over `file://` instead of over http. The cross origin headers are only sent by `serve.py`. |
| The model output is empty in the app | The `tokenizer.json` in the assets is from a different run than the graphs. Rebuild the demo. |

---

## 12. A ten minute smoke test

The shortest path that touches every module.

```bash
cd ~/Documents/Projects/mangokeyboard
source .venv/bin/activate

python -m pytest tests -q                                  # unit tests
python -m scripts.demo --onnx runs/demo/onnx-int8 \
    --tokenizer runs/ml_mini/tokenizer.json --word ente     # python runtime
python -m scripts.build_demo --checkpoint runs/ml_mini/best.pt --quantize
python -m scripts.check_demo --url http://localhost:8000   # browser runtime
cd runtime && ./gradlew test                               # kotlin runtime
cd .. && ./runtime/gradlew -p runtime :app:assembleDebug   # android app
adb install -r runtime/app/build/outputs/apk/debug/app-debug.apk
```

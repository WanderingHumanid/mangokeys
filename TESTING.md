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
# The word list of the keyboard, with the count of every word
python -m data.scripts.download_frequency --out data/raw/frequency
python -m data.scripts.download_dakshina_lexicon --out data/raw/dakshina
python -m scripts.build_lexicon --limit 0 --distance 1 --out data/processed/lexicon

# The character n-gram language model that ranks the candidates
python -m scripts.train_ngram --max-words 400000 --out data/processed/ngram.txt
```

The word list is the commonest Malayalam words from two public sources, the
frequency list of the subtitles and the Malayalam lexicon of the Dakshina
dataset. It holds 68,000 words and it knows 54% of the words in the
Aksharantar test set. The `--limit` is the number of *corpus* words to add on
top, and the corpus is not worth adding: it does not move the coverage.

The `--distance` is how far the lookup looks from a word. One edit is enough
and it makes the index four times smaller.

The language model is a single text file. It is measured in
`scripts/eval_suggestions.py` and it is off by default, because it did not
beat the corpus frequency.

---

## 3. Training

```bash
python -m model.train --config configs/ml_mini.yaml
```

This trains on a GPU when one is visible. The run prints the loss every 100
steps and the validation numbers after every epoch. The best checkpoint lands
in `runs/ml_mini/best.pt` and the history in `runs/ml_mini/history.json`.

There is a second model, the **prefix model**. It is trained on the same words
plus the pairs where the roman side is cut short, so it learns to stop in the
middle of a word and the word list can complete it.

```bash
# The prefix training pairs
python -m data.scripts.build_prefix_pairs --out data/processed/train_prefix.tsv \
    --fraction 0.4 --word-fraction 0.25

# The training run itself
python -m model.train --config configs/ml_mini_prefix.yaml \
    --tokenizer runs/ml_mini/tokenizer.json
python -m model.export_onnx --checkpoint runs/ml_prefix/best.pt --out runs/prefix/onnx
python -m model.quantize --onnx runs/prefix/onnx --out runs/prefix/onnx-int8
```

A full run takes a few hours on a laptop GPU. The two models are compared in
`scripts/compare_models.py`.

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
python -m scripts.eval_suggestions --onnx runs/demo/onnx-int8

# The two models against each other
python -m scripts.compare_models
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
# Assemble the demo folder from the committed model bundle: the graphs, the
# tokenizer, the lexicon and the runtime of the browser
python -m scripts.build_web --out dist/web

# Serve it. The headers are what let the wasm runtime use four threads
python dist/web/serve.py --port 8000
```

Open `http://localhost:8000`. The page is the keyboard. Type in the box above
it and the candidate strip fills up, the first candidate is the one the space
bar takes, and the space bar takes it in place, in the field.

What to check, in order:

1. The page is dark and the keyboard looks like the one on the phone.
2. Typing `ente` shows `എൻ്റെ` first in the strip, on the accent colour.
3. The space bar replaces `ente` with `എൻ്റെ` in the field and adds a space.
4. Clicking another candidate replaces the word with that one.
5. The keys on the page type into the field, the backspace deletes in it and
   the 123 key opens the numbers.
6. The key you press on your own keyboard lights up on the page.
7. The table under the keyboard has the three measured systems and the note
   about the coverage of the word list.
8. The page does not stutter while you type.

To check it from another machine on the same network, use the address of this
one and make sure the phone or the other computer is on the same network:

```bash
hostname -I
```

There is a headless check as well, which is what the build runs. It types a
word, presses the space bar, taps a candidate, types with the keys on the
page and opens the numbers layer:

```bash
python -m scripts.check_demo --url http://localhost:8000 \
    --shot .impeccable/review/desktop.png \
    --shot-mobile .impeccable/review/mobile.png
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

The app is an `InputMethodService`. It carries two models, the word list and
the language model in its assets, and it asks for no permissions.

### 9.1 Build the debug package

```bash
cd runtime
export ANDROID_HOME=~/Android/Sdk
./gradlew :app:assembleDebug
```

The packages land in `apps/android/build/outputs/apk/debug/`, one for each
processor and one for all of them. The universal one installs on any device.
The build copies the assets out of `runs/` and `data/processed/`, so the model,
the prefix model, the word list and the language model have to be built first
with steps 3 and 5.

The paths can be changed:

```bash
./gradlew :app:assembleDebug \
    -PmodelDir=../runs/demo/onnx-int8 \
    -PprefixModelDir=../runs/prefix/onnx-int8 \
    -PlexiconDir=../data/processed/lexicon \
    -PngramFile=../data/processed/ngram.txt
```

### 9.2 Run it on the emulator

There is an AVD called `Pixel_10` already set up.

```bash
# Start the emulator in the background
emulator -avd Pixel_10 -no-snapshot -no-audio -gpu swiftshader_indirect &
```

Wait for the boot to finish. This takes two or three minutes:

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
adb install -r apps/android/build/outputs/apk/debug/app-universal-debug.apk
```

The app copies the assets into its own directory the first time it starts.
When the assets change, clear the app first, otherwise the old ones are kept:

```bash
adb shell pm clear com.mangokeyboard
```

### 9.5 Turn the keyboard on

Android will not let an app register a keyboard on its own, so this has to be
done once by hand. It can also be done from the command line, and that is
what the checks below use:

```bash
adb shell ime enable com.mangokeyboard/.ManglishKeyboardService
adb shell ime set com.mangokeyboard/.ManglishKeyboardService
adb shell ime list -s -a | grep manglish
```

The last command has to print
`com.mangokeyboard/.ManglishKeyboardService`. If it prints nothing, the
service was not registered, and the reason is in the log:

```bash
adb logcat -d | grep -i "Skipping input method"
```

The service has to ask for the `BIND_INPUT_METHOD` permission, otherwise
Android skips it. That permission is a system one that only the platform can
hold, so the app still asks the user for nothing.

On a phone the same thing is done by hand: **Settings → System → Languages
and input → On-screen keyboard → Manage keyboards**, then pick **Manglish
Keyboard** and turn it on.

### 9.6 Test it

The keyboard is called **MangoKeys**. It behaves like an ordinary keyboard:
every key goes straight into the text field, so the backspace, the cursor and
the selection belong to the application. The suggestion strip above the keys
holds the Malayalam, and tapping one replaces the word that is being typed.

Open any text field. The browser address bar is the easiest one to reach:

```bash
adb shell am start -a android.intent.action.VIEW -d "https://example.com"
```

What to check, in order:

1. The keyboard appears when a text field is focused.
2. The layout is the standard one: three rows of letters, the shift on the
   left and the backspace on the right, and 123, comma, space, period and
   enter on the last row.
3. Typing `ente` puts `ente` in the field, in Manglish, and shows `എൻ്റെ`
   first in the strip.
4. Tapping `എൻ്റെ` replaces `ente` in the field with `എൻ്റെ`.
5. The backspace deletes in the field, not in the keyboard.
6. The space bar commits the top suggestion and then a space.
7. The 123 key shows the numbers and the abc key goes back.
8. Typing `aash` shows `ആശംസകൾ`, because the prefix model stops in the
   middle of the word and the word list completes it.
9. The globe key opens the input method picker.
10. The strip and the keys follow the theme, in the dark theme as well.

The suggestions can be read from the log, which is easier than looking at the
screen:

```bash
adb logcat -s MangoKeys
```

Each line is the suggestion strip after a keystroke:

```
I MangoKeys: suggestions: എൻ്റെ | എൻറെ | എന്റെ | നൈറ്റ്
```

### 9.7 The application page

The application of the keyboard has a test bench at the top, so that the
keyboard can be tried without switching the input method, and the settings
under it.

```bash
adb shell am start -n com.mangokeyboard/.MainActivity
```

The test bench is a text field, a line under it that shows what the last word
would become, and a row of example words. Tap an example and the line fills
in with the Malayalam. The models are loaded in the background when the page
opens, so the first answer takes a moment.

Under it are the settings: the dark theme, six accent colours and a custom
colour picker with three sliders, and the behaviour switches.

The dark theme is not only the keyboard. The whole page follows it: the
background, the cards, the field, the text and the status bar. Turn it on and
the page is rebuilt in the dark theme at once.

The settings are stored in the preferences of the app:

```bash
adb shell run-as com.mangokeyboard cat shared_prefs/mangokeys.xml
```

The logs of the keyboard are visible with:

```bash
adb logcat -s MangoKeys
```

To watch them while you type, leave that running in a second terminal.

### 9.8 Check that it asks for nothing

```bash
# The permission list must be empty
adb shell dumpsys package com.mangokeyboard | grep -A 20 "requested permissions"
```

A keyboard that asks for the *full access* permission shows a warning dialog
to the user. This one declares no permissions at all, so the list is empty.

### 9.9 Measure the latency on the device

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
| The keyboard is not in `ime list -s` | The service did not ask for the `BIND_INPUT_METHOD` permission. The log says `Skipping input method`. |
| The keyboard does not show up when a field is focused | The input method was not selected. Run `adb shell ime set com.mangokeyboard/.ManglishKeyboardService`. |
| The suggestions are the old ones | The app kept the old assets. Run `adb shell pm clear com.mangokeyboard` and open the app again. |
| `adb: no devices found` | The phone is not in the USB debugging list, or the prompt was dismissed. Run `adb kill-server && adb start-server` and replug it. |
| The phone shows `unauthorized` | Accept the prompt on the phone screen. If it does not appear, revoke the USB debugging authorisation in the developer options and replug. |
| The emulator is very slow | It is running without hardware acceleration. Check that `/dev/kvm` exists and is readable. |
| The suggestion strip is empty | The assets did not make it into the package. Run `unzip -l apps/android/build/outputs/apk/debug/app-universal-debug.apk | grep assets`. |
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
python -m scripts.build_web --out dist/web                 # the demo folder
python -m scripts.check_demo --url http://localhost:8000   # browser runtime
cd runtime && ./gradlew test                               # kotlin runtime
cd .. && cd runtime && ./gradlew :app:assembleDebug        # android app
adb install -r apps/android/build/outputs/apk/debug/app-universal-debug.apk
adb shell ime enable com.mangokeyboard/.ManglishKeyboardService
adb shell ime set com.mangokeyboard/.ManglishKeyboardService
adb shell am start -a android.settings.SETTINGS             # the keyboard on the device
adb logcat -s MangoKeys                              # what it suggests
```

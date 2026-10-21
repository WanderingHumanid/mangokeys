# MangoKeys v0.01

The first public build of MangoKeys, a Manglish to Malayalam keyboard.

## What is in this release

- An Android input method that writes Malayalam as you type. Every key goes
  into the text field that has the focus, so the backspace, the cursor and
  the selection are the ones of the app you are typing in.
- A suggestion strip above the keys. The word you are typing is translated
  and the candidates sit above the keys. Tapping one replaces the word in
  the field. The space bar commits the first candidate.
- Two character level models of 5.37 million parameters each, running on the
  phone. One transliterates whole words. The second one was trained on the
  same words plus the pairs where the roman side is cut short, so it knows
  where the word is going, and the word list completes it.
- A word list of 68 thousand Malayalam words with a deletion index, so the
  keyboard can find the real words around what the model guessed.
- A dark and a light theme, six accent colours and a custom colour picker.
- No permissions. No network. Nothing you type leaves the phone.

## How to install

1. Download `mangokeys-<version>.apk` from this page.
2. Open it on your phone and allow the install from this source.
3. Open the keyboard settings of Android, pick MangoKeys under the list of
   keyboards.
4. Open MangoKeys once and turn the dark theme on if you want it.

The APK is a prototype. It is signed with a development key, so Android will
warn you about the unknown source the first time. Nothing else is wrong with
it.

## The web version

The same model runs in the browser, with nothing uploaded and no server. The
`mangokeys-web-demo.zip` file on this page is the whole demo. Unpack it and
serve the folder with any static server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## What it is not yet

The word list covers the common words well and the tail of the corpus badly.
The language model is in the code but it is off by default, because it
measured as no help. See the report in the repository for the numbers and for
what we would do next.

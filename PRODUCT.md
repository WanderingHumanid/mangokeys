# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

The keyboard itself is an Android input method, so the same product has two
surfaces: the native app and the browser demo. The browser demo is the one
this file describes. The two share the model, the word list and the ranking,
and the demo is the web presentation of the app.

## Stack

Plain static HTML, CSS and JavaScript. No framework, no build step for the
page itself. The model is ONNX Runtime Web in a Web Worker. The runtime of
the browser is a module in `runtime/js`, copied into the demo folder by
`scripts/build_web.py`. Served by `apps/expo-web/serve.py`, which sends the
cross origin isolation headers that let the WebAssembly runtime use more than
one thread. Published by the GitHub Pages workflow.

## Users

Malayalam speakers who type in English letters on their phones, and the
students and reviewers who want to see the model work without installing an
APK. The primary job of the first group is to write a message in their own
language without switching keyboards or learning a new script.

## Product Purpose

MangoKeys is a Manglish to Malayalam input method. The user types the word
the way they type it, in Latin letters, and the keyboard offers the Malayalam
spellings of the words they might have meant. Success is a word that is
offered first time, and a suggestion strip that never appears empty.

## Positioning

The model does not transliterate a finished word, it ranks the words the user
might have meant, because the romanization of Malayalam hides distinctions
that the orthography makes. The contribution of the project is that the
training data has to look the way people really type, not the way the
corpus spells things, and that gap is measured rather than assumed.

## Operating Context

The demo is opened on a phone more often than on a laptop, and it is opened
without an internet connection, since the model runs in the browser and
nothing is uploaded. The user types one word at a time and reads the
suggestions in Malayalam script, which they can read but might not type. The
page is used as a demonstration of the model in a college project review as
much as it is used as a keyboard.

## Capabilities and Constraints

- The model is 5.37 million parameters, quantized to int8, and it is 12 MB
  of payload with the word list. The page must be usable on a mid range
  phone, so the model runs off the main thread and the rules answer first.
- Nothing is uploaded. There is no server and no account.
- The suggestion strip is filled from the rules and the word list in about
  three milliseconds, and the model's answer refines it a moment later.
- The space bar commits the first suggestion. Tapping any suggestion
  replaces the word in the field.
- The theme is always dark. This was a decision of the user, made against a
  recommendation to follow the system preference, because the dark keyboard
  is the app's strongest look.
- The page holds the keyboard and a compact results section. The full
  benchmark table stays in the repository.

## Brand Commitments

The name is MangoKeys, in both surfaces, including the name of the input
method in the system keyboard switcher. The voice is plain and technical. No
emojis anywhere, and no long dashes. The demo borrows the visual language of
the Android app, because the point of the demo is to show the app.

## Evidence on Hand

The measured results are in `evals/results`, and the report is `REPORT.md`.
The numbers the page shows come from `apps/expo-web/models/eval.json`,
which is written by `scripts/make_report.py`. There are no testimonials and
no user numbers, and none may be invented. The word list covers 54% of the
words in the test set and 97% of the common words, which is the honest limit
of what the demo can promise.

## Product Principles

1. The suggestion strip is never empty. The rules and the word list answer
   before the model does, and the keyboard feels instant.
2. Show the measurement. The results section is short and every number in it
   came from a script in this repository.
3. The demo is the app. Anything that can be demonstrated on the web should
   be demonstrated the way the phone does it.
4. Nothing leaves the device. There is no server and no telemetry, and the
   page says so.

## Accessibility and Inclusion

The Malayalam script is the product, so the suggestions are rendered in a
font that can shape Malayalam conjuncts. The interface is a dark surface
with a light text, which is the contrast ratio the app already uses.
Keyboard focus is visible and the suggestion buttons are real buttons, so
they can be reached with a keyboard.

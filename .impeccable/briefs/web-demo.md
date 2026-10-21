# Surface brief: the MangoKeys web demo

Development only. Nothing in this file ships to the browser.

## Scope and visitor mode

The surface is the web demo at `apps/expo-web`. The visitor mode is Operate:
the visitor came to type a word in Manglish and see the Malayalam. The page
is a working keyboard first and a demonstration second.

## Audience, job, proof

- Audience: Malayalam speakers who type in Latin letters, and reviewers of
  the college project who want to see the model work without installing an
  APK.
- Job: type one word, read the Malayalam, and see that the suggestion is the
  right one.
- Action: type in the compose field, press space, or tap a candidate.
- Proof: the measured results, compact, below the keyboard. Every number
  comes from `evals/results` through `scripts/make_report.py`.
- Constraints: nothing uploaded, no server, works offline, 12 MB of model
  payload, the model must not block the main thread.

## The world

The visual world is the MangoKeys Android app. It is settled and it is not
being reinvented here. The web demo inherits its palette, its density, its
type and its one signature move, and it copies its layout, because a demo
that does not look like the app is not a demo of the app.

The app contributes: the dark palette (#1B1D21 body, #23262B panel, #31353B
key, #3A3F46 function key, #E8EAED text), the Ocean accent (#2C5F8A) as the
default, the platform sans for the interface and the platform Malayalam face
for the script, the compact density of a 44 pixel key, and the signature
move: a candidate strip in which the first candidate is the accent pill and
the space bar commits it.

The world supplies type, palette, density and the signature move. It does
not supply the layout of the page, and the page is laid out as a web page:
a header, the compose area, the keyboard, the results, the footer.

## Memorable moment

The page behaves like the keyboard it is. The candidate strip sits where the
keyboard's strip sits, the space bar commits the highlighted candidate, and
the key you press on your own keyboard lights up on the on-screen one. The
word in the compose field changes from Manglish to Malayalam in place, the
way it does on the phone.

## Direction contract

THESIS. The demo is the keyboard, on the web, with the real model running
behind it. The page refuses the demo-page arrangement of a headline, a
sentence about the project and a form in the middle.

OWN-WORLD. The exact palette of the Android keyboard, the platform sans, a
compact 44 pixel key, the accent pill for the first candidate, drawn icons in
one stroke weight for shift, backspace and enter.

STORY. The visitor types one word and the page answers in Malayalam, in the
place the word was typed, and then shows the measurement that says how often
that answer is right.

FIRST VIEWPORT. The wordmark and the model state at the top, the compose
field with its example words, then the keyboard: the candidate strip, the
divider, the four rows of keys. The keyboard takes the middle of the first
screen and it is the largest object on it.

FORM. The app's own layout, extended to a web page. Code led.

FINISH. unreviewed and undocumented is unfinished; this build ends with the
finish review, the verdict, DESIGN.md, and every shipping raster carrying
its provenance.

## Unresolved

- The numbers layer with a key value cache is still the biggest cost in the
  mobile runtime, and the browser feels it.
- The word list covers the common words and not the tail, so the demo will
  be wrong on the named entities. The page should say so rather than hide
  it.

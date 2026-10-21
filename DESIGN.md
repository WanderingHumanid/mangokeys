---
name: MangoKeys
description: A Manglish to Malayalam keyboard, on the phone and in the browser.
colors:
  ground: "#101215"
  body: "#1b1d21"
  surface: "#23262b"
  key: "#31353b"
  function: "#3a3f46"
  function-text: "#c9ced5"
  warning: "#e79a9a"
  line: "#2b2f35"
  line-soft: "#24272c"
  text: "#e8eaed"
  muted: "#a2a9b2"
  faint: "#8b939d"
  accent: "#2c5f8a"
  accent-bright: "#7fb0de"
typography:
  display:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  headline:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.5
  title:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  native:
    fontFamily: "Noto Sans Malayalam, Nirmala UI, Kartika, system-ui, sans-serif"
    fontSize: "21px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.5
  key:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1
  key-large:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1
  key-small:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1
  candidate:
    fontFamily: "Noto Sans Malayalam, Nirmala UI, Kartika, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.35
rounded:
  sm: "10px"
  md: "12px"
  lg: "14px"
  panel: "20px"
  panel-small: "16px"
  pill: "18px"
  scroll: "2px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "14px"
  lg: "20px"
  xl: "40px"
components:
  key:
    backgroundColor: "{colors.key}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    height: "44px"
  key-function:
    backgroundColor: "{colors.function}"
    textColor: "{colors.faint}"
    rounded: "{rounded.sm}"
    height: "44px"
  key-accent:
    backgroundColor: "{colors.accent}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    height: "44px"
  candidate:
    backgroundColor: "{colors.function}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
  candidate-first:
    backgroundColor: "{colors.accent}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    padding: "6px 18px"
  chip:
    backgroundColor: "{colors.key}"
    textColor: "{colors.muted}"
    rounded: "999px"
    padding: "4px 10px"
  field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: "12px 14px"
---

# Design System: MangoKeys

## Overview

**Creative North Star: "The keyboard on the table"**

MangoKeys is one product in two places: an Android input method and a web
demo. They share a model, a word list and a ranking, and they share a face.
The web demo is not a page that describes the keyboard, it is the keyboard,
drawn on the page, doing the same thing with the same parts in the same
arrangement.

The world is settled and it is the world's of the app: a near-black ground,
four steps of cool neutral lifted by small lightness changes rather than by
borders, one accent that does real work, and a density that matches a phone.
Nothing is decorated. Every panel is a step lighter than the ground it sits
on, every key is a step lighter than the keyboard it sits on, and the only
colour on the screen is the accent, which marks the one candidate the space
bar will take and the one key that commits it.

The voice is plain and technical, the way the report is written. There are no
emoji, no long dashes, and no claim on the page that a script in the
repository did not produce.

**Key Characteristics:**

- One ground, four lifts, one accent.
- The accent is a verb: it marks the action the space bar will take.
- Every number on the page is read from a file that an evaluation wrote.
- The Malayalam is the largest type on the page.
- Drawn icons, one stroke weight, never a glyph standing in for a symbol.

## Colors

A cool near-black ground with four lifts above it, and one accent that is
allowed to be the only saturated thing on the screen.

### Primary

- **Ocean** (`#2c5f8a`): the accent of the app, and the default. It fills the
  first candidate, the enter key, the selected switch and the committed word.
  It is never used for text on the ground, where it would not be readable.

### Neutral

- **Ground** (`#101215`): the page behind everything. The darkest value in
  the system.
- **Body** (`#1b1d21`): the keyboard panel.
- **Surface** (`#23262b`): the candidate strip and the compose field.
- **Key** (`#31353b`): the letter keys, and the example chips of the test
  bench in the app.
- **Function** (`#3a3f46`): the keys that do something other than type a
  letter, and the candidates that are not the first one.
- **Function text** (`#c9ced5`): the label of a function key, a shade quieter
  than the letters.
- **Warning** (`#e79a9a`): the state of the model when something failed.
- **Line** (`#2b2f35`): the divider under the strip.
- **Line soft** (`#24272c`): the hairlines inside the results table.
- **Text** (`#e8eaed`): everything that is being read.
- **Muted** (`#a2a9b2`): the supporting text, the chip labels.
- **Faint** (`#8b939d`): the hints and the captions, and the label of the
  space bar when there is nothing to take.

### Named Rules

**The One Accent Rule.** The accent appears on at most three things at once:
the first candidate, the enter key and the state of the switch. Its rarity is
what makes it mean "this one".

**The Lightness Ladder Rule.** A panel is separated from what is behind it
by lightness, not by a border. The ladder is ground, body, surface, key,
function. A hairline is used only where a ladder step would be ambiguous.

**The Contrast Floor.** Body text and placeholders sit at 4.5:1 or better
against the surface they are on, and the faintest tone in the system is
`#8b939d`, which clears that on every surface it is used on.

## Typography

**Display Font:** the platform sans (system-ui, Segoe UI, Roboto, Noto Sans)
**Body Font:** the platform sans
**Label/Mono Font:** none. The numbers in the table are the platform sans
with tabular numerals, which is the only way to make a column of percentages
read as a column.

**Character:** the lettering is the platform's, because the app uses the
platform's, and the demo that claims to be the app has to look like it. The
only typeface with a personality here is the Malayalam one, and it is used for
the only thing the product is about: the Malayalam itself.

### Hierarchy

- **Display** (600, 19px, -0.015em): the wordmark. One word, one line.
- **Headline** (600, 14px): the heading of the results section.
- **Title** (400, 15px): the page body.
- **Body** (400, 15px, 1.5): the explanatory lines, the note under the table.
- **Native** (400, 21px, 1.45): the Malayalam in the compose field. The
  largest type on the page, because it is the thing the visitor came for.
- **Label** (400, 12.5px): the state of the model, the captions, the table
  headers. Never uppercase, never tracked.
- **Key** (400, 17px, 18px on a large screen, 16px on a narrow one): the
  letter on a key.
- **Candidate** (400, 17px, 1.35): the word in the strip, in the Malayalam
  face.

### Named Rules

**The Native Rule.** The Malayalam is always set in the Malayalam face and
never in the interface face, even when a single word appears inside a
sentence.

**The Plain Label Rule.** Labels are sentence case at their own size. There
are no small caps, no tracked caps and no kickers.

## Layout

One column, centred, that holds the product the way a phone holds it.

| Width | Column | Key height | Key size |
|---|---|---|---|
| below 640px | full, 12px padding | 44px | 16px |
| 640 to 1099px | 600px | 44px | 17px |
| 1100px and above | 720px | 54px | 18px |

The column grows and the keyboard grows with it, and the proportions of the
keys stay the same. The order down the page never changes: the wordmark and
the state of the model, the compose field, the keyboard with its strip, the
measured results, the footer.

The vertical rhythm is a multiple of four, with more space above a heading
than below it. The gap between the compose field and the keyboard is 18px,
between the keyboard and the results 40px, and between the results and the
footer 44px.

The results table is a real table: a header row, a hairline under it, right
aligned columns of tabular numerals at 84px each.

## Elevation & Depth

The system is flat and layered. Depth is made of lightness, not of shadows.
There is exactly one shadow in the whole system, under the keyboard panel,
because it is the one object that floats above the page:

- **Panel** (`0 12px 28px -18px rgba(0, 0, 0, 0.85)`): the keyboard, and
  nothing else.

Everything else is a step of the lightness ladder or a hairline.

### Named Rules

**The One Shadow Rule.** A shadow means the object floats. Only the keyboard
floats. A card, a chip, a field or a table never does.

## Shapes

The corner language is soft but not round. The keys are 10px, the fields and
the chips 12px, the compose panel 14px, the keyboard 20px and 16px on a
narrow screen, and the scroll bar 2px. The first
candidate is a pill at 18px, because it is the one thing on the page that is
alone in its row.

The keys have no border. The fields have a 1px hairline in the soft line
colour. The strip has a 1px divider in the function colour, which is what
separates the candidates from the keys.

## Components

### Keys

- **Shape:** 10px radius, no border, 44px tall (54px on a large screen).
- **Letter:** the key colour, the text colour, 17px.
- **Function:** the function colour, the faint colour, 14px. The shift, the
  backspace, the 123, the comma and the period.
- **Accent:** the enter key and the space bar, when there is something to
  take. The accent is on the space bar only while a candidate exists.
- **Hover / Focus:** a step lighter on hover, a 1px lift on press, and the
  focus ring in the bright accent.
- **Icons:** the shift, the backspace and the enter are drawn as SVG in one
  stroke weight, never as a glyph from a font.

### Candidates

- **Shape:** 12px radius, 6px by 14px of padding, the Malayalam face at 17px.
- **First:** the accent fill, white text, 18px radius and 18px of side
  padding. It is the one the space bar takes.
- **Others:** the function colour, so that they read as candidates and not as
  keys.
- **Overflow:** the row scrolls sideways, and the edge fades only while there
  is more to scroll to.

### Chips

- **Style:** the key colour, the muted colour, a pill radius, 13px.
- **State:** the example chips of the test bench in the app fill its field
  when they are tapped. The web page has none. The visitor types.

### Inputs

- **Style:** the surface colour, a 1px soft hairline, 14px radius, 12px by
  14px of padding.
- **Focus:** the caret is the bright accent and the panel takes the accent
  border for the length of the commit.
- **Content:** the Malayalam face at 21px, because the field is where the
  answer lands.

### The results table

- **Shape:** no frame, no fill. A header row in the faint colour with a
  hairline under it, rows with a soft hairline between them and none under
  the last.
- **Numbers:** tabular, right aligned, 84px wide, the text colour.

### Navigation

There is none. The footer carries three links, the repository, the app and
the report. They are the muted colour, they have no underline at rest, and
they take the bright accent underline on hover.

## Do's and Don'ts

### Do:

- **Do** keep the candidate strip, the keyboard and the compose field in that
  order, with the strip directly above the keys.
- **Do** let the accent mark the first candidate and the enter key, and
  nothing else on the screen at the same time.
- **Do** read every number from a file that an evaluation wrote, and say so
  in the page when the number has a limit.
- **Do** set the Malayalam in the Malayalam face, at the largest size it is
  given on the page.
- **Do** draw the icons.

### Don't:

- **Don't** add a second accent, a gradient, a glow or a glass effect. The
  ground is the product.
- **Don't** put the strip's candidates in the key colour. They would read as
  keys and the signature would be lost.
- **Don't** use a glyph font for the shift, the backspace or the enter.
- **Don't** uppercase the labels, and don't put a small tracked label above
  a heading.
- **Don't** claim anything the repository does not measure, including the
  coverage of the word list.

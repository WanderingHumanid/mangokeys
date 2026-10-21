/**
 * The demo.
 *
 * It behaves like the keyboard on the phone. The word being typed is
 * translated, the candidates sit in the strip above the keys, the space bar
 * commits the first one and tapping any of them replaces the word in the
 * field. The model runs in a worker, and the rules and the word list answer
 * on the main thread while it thinks, so the strip is never empty.
 */

import { Lexicon } from "./lexicon.js";
import { SuggestionEngine } from "./suggest.js";
import { OnScreenKeyboard } from "./keyboard.js";

const elements = {
  status: document.getElementById("status"),
  field: document.getElementById("field"),
  strip: document.getElementById("strip"),
  keys: document.getElementById("keys"),
  examples: document.getElementById("examples"),
  benchBody: document.querySelector("#bench-table tbody"),
  benchNote: document.getElementById("bench-note"),
  clock: document.getElementById("clock"),
};

const state = {
  lexicon: null,
  ranker: null,
  worker: null,
  candidates: [],
  requestId: 0,
  timer: null,
};

const EXAMPLE_WORDS = [
  "ente peru",
  "njangalkum",
  "sundari",
  "keralathil",
  "sughamano",
  "ningalkkum",
];

const keyboard = new OnScreenKeyboard(elements.keys, {
  onCharacter: (letter) => type(keyboard.currentLetter(letter)),
  onSpace: () => typeSpace(),
  onBackspace: () => typeBackspace(),
  onEnter: () => typeEnter(),
});

// ---------------------------------------------------------------- the field

/** A word is a run of Roman letters. The Malayalam in the field is not one. */
function isWordChar(character) {
  return /[A-Za-z'\-]/.test(character);
}

/** The word the caret is in. */
function currentWord() {
  const value = elements.field.value;
  const caret = elements.field.selectionStart ?? value.length;
  let start = caret;
  while (start > 0 && isWordChar(value[start - 1])) start -= 1;
  return value.slice(start, caret);
}

/** Put a word where the word being typed is, in the field itself. */
function replaceWord(word) {
  const field = elements.field;
  const caret = field.selectionStart ?? field.value.length;
  const before = field.value.slice(0, caret);
  const after = field.value.slice(caret);
  let start = caret;
  while (start > 0 && isWordChar(before[start - 1])) start -= 1;
  field.value = `${before.slice(0, start)}${word}${after}`;
  const position = start + word.length;
  field.setSelectionRange(position, position);
  flashWord();
}

/** The word that was just committed lights up for a moment. */
function flashWord() {
  const compose = elements.field.closest(".compose");
  compose.classList.remove("committed");
  void compose.offsetWidth;
  compose.classList.add("committed");
  window.setTimeout(() => compose.classList.remove("committed"), 520);
}

// ------------------------------------------------------------------ typing

function type(letter) {
  const field = elements.field;
  const caret = field.selectionStart ?? field.value.length;
  field.value = field.value.slice(0, caret) + letter + field.value.slice(caret);
  field.setSelectionRange(caret + letter.length, caret + letter.length);
  keyboard.consumeShift();
  refresh();
}

/** The candidate the space bar takes. It is the first one in the strip. */
function firstSuggestion() {
  const first = elements.strip.querySelector(".candidate");
  return first ? first.textContent : null;
}

function typeSpace() {
  const word = currentWord();
  const best = firstSuggestion();
  if (word && best) replaceWord(best);
  const field = elements.field;
  const caret = field.selectionStart ?? field.value.length;
  field.value = `${field.value.slice(0, caret)} ${field.value.slice(caret)}`;
  field.setSelectionRange(caret + 1, caret + 1);
  refresh();
}

function typeBackspace() {
  const field = elements.field;
  const caret = field.selectionStart ?? field.value.length;
  if (caret === 0) return;
  field.value = field.value.slice(0, caret - 1) + field.value.slice(caret);
  field.setSelectionRange(caret - 1, caret - 1);
  refresh();
}

function typeEnter() {
  const field = elements.field;
  const caret = field.selectionStart ?? field.value.length;
  field.value = `${field.value.slice(0, caret)}\n${field.value.slice(caret)}`;
  field.setSelectionRange(caret + 1, caret + 1);
  refresh();
}

// ------------------------------------------------------------- suggestions

/**
 * Update the strip.
 *
 * The rules and the word list answer straight away, on this thread. The
 * model's answer comes from the worker a moment later and refines the strip.
 */
function refresh() {
  const word = currentWord();
  if (!word) {
    state.candidates = [];
    showStrip([]);
    keyboard.setSpaceActive(false);
    return;
  }

  const instant = state.ranker ? state.ranker.suggest(word) : [];
  showStrip(instant.map((item) => item.text));
  requestModel(word);
}

function requestModel(word) {
  if (!state.worker) return;
  state.requestId += 1;
  state.worker.postMessage({ id: state.requestId, roman: word, nBest: 5 });
}

function showStrip(words) {
  elements.strip.innerHTML = "";
  if (!words.length) {
    // The strip of a keyboard is never empty. On the phone it holds the
    // icons, so here it holds the one thing the visitor needs to know.
    const hint = document.createElement("span");
    hint.className = "strip-hint";
    hint.textContent = "the space bar takes the first one";
    elements.strip.appendChild(hint);
    keyboard.setSpaceActive(false);
    return;
  }
  for (const [index, word] of words.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === 0 ? "candidate candidate-first" : "candidate";
    button.textContent = word;
    button.addEventListener("click", () => {
      replaceWord(word);
      refresh();
    });
    elements.strip.appendChild(button);
  }
  keyboard.setSpaceActive(true);
  // The edge only fades when there is more to scroll to.
  elements.strip.classList.toggle(
    "overflowing",
    elements.strip.scrollWidth > elements.strip.clientWidth + 1,
  );
}

// ----------------------------------------------------------------- the model

function startWorker() {
  const worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = (event) => {
    const { id, candidates, scores, prefix, error, ready, threads, ms } = event.data;
    if (error) {
      elements.status.textContent = `the model failed: ${error}`;
      elements.status.classList.add("bad");
      return;
    }
    if (ready) {
      elements.status.textContent = `model ready on ${threads} threads`;
      elements.status.classList.remove("bad");
      return;
    }    if (id !== state.requestId) return;

    // The completions come from the prefix model and the word list.
    let completion = [];
    if (prefix && state.ranker) {
      completion = state.ranker.lexicon.complete(prefix, 8);
    }
    const ranked = state.ranker
      ? state.ranker.suggest(currentWord(), candidates ?? [], scores ?? [], completion)
      : [];
    state.candidates = ranked.map((item) => item.text);
    showStrip(state.candidates);

    if (typeof ms === "number") {
      elements.clock.textContent = `${Math.round(ms)} ms`;
    }
  };
  state.worker = worker;
  return worker;
}

// ------------------------------------------------------------------- boot

async function boot() {
  buildExamples();
  wireField();
  wirePhysicalKeyboard();

  state.worker = startWorker();
  // The worker loads the model when it gets its first message, so it is
  // asked for one now, with nothing to translate.
  state.worker.postMessage({ id: 0, roman: "", nBest: 0 });

  // The word list is loaded on this thread as well, so that the rules and
  // the list can answer before the model has finished thinking.
  try {
    state.lexicon = await Lexicon.load("./models/lexicon");
    state.ranker = new SuggestionEngine(state.lexicon, null);
  } catch (error) {
    elements.status.textContent = "the word list failed to load";
    elements.status.classList.add("bad");
  }
  refresh();
  loadBenchmarks();
}

function buildExamples() {
  for (const word of EXAMPLE_WORDS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    button.textContent = word;
    button.addEventListener("click", () => {
      elements.field.value = word;
      elements.field.setSelectionRange(word.length, word.length);
      refresh();
    });
    elements.examples.appendChild(button);
  }
}

function wireField() {
  elements.field.addEventListener("input", () => {
    window.clearTimeout(state.timer);
    state.timer = window.setTimeout(refresh, 60);
  });
  elements.field.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      // The space bar commits the highlighted candidate, the way it does on
      // the phone. The browser must not also insert a space.
      event.preventDefault();
      typeSpace();
    }
  });
  elements.field.addEventListener("click", () => {
    window.setTimeout(refresh, 0);
  });
}

/** The keys of the physical keyboard light up on the on-screen keyboard. */
function wirePhysicalKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key;
    if (key === " ") {
      keyboard.flash("space");
      return;
    }
    if (key === "Backspace") {
      keyboard.flash("backspace");
      return;
    }
    if (key === "Enter") {
      keyboard.flash("enter");
      return;
    }
    if (key.length === 1) keyboard.flash(key.toLowerCase());
  });
}

// ---------------------------------------------------------------- the proof

async function loadBenchmarks() {
  try {
    const response = await fetch("./models/eval.json");
    if (!response.ok) throw new Error("missing");
    renderBenchmarks(await response.json());
  } catch {
    elements.benchNote.textContent =
      "The table is built by scripts/make_report.py and was not found.";
  }
}

function renderBenchmarks(data) {
  const entry = data.find((item) => item.file === "test_aksharantar.tsv");
  if (!entry) return;
  // The names in the evaluation file are the names of the runs. The page
  // says what each one is.
  const rows = [
    ["rule engine (ours)", "the rule engine"],
    ["our model", "the model"],
    ["our model with the word list", "the model with the word list"],
  ];
  elements.benchBody.innerHTML = "";
  for (const [name, label] of rows) {
    const result = entry.systems?.[name];
    if (!result) continue;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${label}</td>
      <td class="num">${(result.top1 * 100).toFixed(1)}%</td>
      <td class="num">${(result.cer * 100).toFixed(1)}%</td>
    `;
    elements.benchBody.appendChild(row);
  }

  // The word list is what makes the difference, and it only knows part of
  // the words. The page says so rather than hide it.
  const coverage = entry.dictionary_coverage;
  if (typeof coverage === "number") {
    elements.benchNote.textContent =
      `The word list knows ${Math.round(coverage * 100)}% of the words in the test set ` +
      "and almost all of the common words. The rest is where the keyboard is wrong.";
  }
}

boot();

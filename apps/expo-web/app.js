import * as ort from "./ort/ort.min.mjs";
import { Transliterator, buildTokenizer, normalizeRoman } from "./transliterator.js";

ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;

const MAX_SRC = 48;
const MAX_TGT = 64;
const BEAM = 4;

const elements = {
  status: document.getElementById("status"),
  word: document.getElementById("word"),
  modelOutput: document.getElementById("model-output"),
  ruleOutput: document.getElementById("rule-output"),
  candidates: document.getElementById("candidate-list"),
  benchBody: document.querySelector("#bench-table tbody"),
  benchNote: document.getElementById("bench-note"),
};

let engine = null;
let timer = null;
let requestId = 0;

/** A small rule engine, so the page can show what the model adds. */
const CONSONANTS = {
  k: "ക", kh: "ഖ", g: "ഗ", gh: "ഘ", ng: "ങ", ch: "ച", chh: "ഛ",
  j: "ജ", jh: "ഝ", nj: "ഞ", t: "ട", T: "ട", th: "ഠ", D: "ഡ",
  dh: "ഢ", N: "ണ", thh: "ത", n: "ന", p: "പ", ph: "ഫ", b: "ബ",
  bh: "ഭ", m: "മ", y: "യ", r: "ര", l: "ല", v: "വ", sh: "ശ",
  S: "ഷ", s: "സ", h: "ഹ", L: "ള", zh: "ഴ", R: "റ", rr: "റ",
};
const VOWELS = { a: "അ", aa: "ആ", i: "ഇ", ii: "ഈ", u: "ഉ", uu: "ഊ", e: "എ", ee: "ഏ", ai: "ഐ", o: "ഒ", oo: "ഓ", au: "ഔ" };
const SIGNS = { a: "", aa: "ാ", i: "ി", ii: "ീ", u: "ു", uu: "ൂ", e: "െ", ee: "േ", ai: "ൈ", o: "ൊ", oo: "ോ", au: "ൗ" };
const CHILLUS = { "ണ": "ൺ", "ന": "ൻ", "ര": "ർ", "ല": "ൽ", "ള": "ൾ", "ക": "ൿ" };

function ruleConvert(text) {
  const keys = (table) => Object.keys(table).sort((a, b) => b.length - a.length);
  const vowelKeys = keys(VOWELS);
  const signKeys = keys(SIGNS);
  const consonantKeys = keys(CONSONANTS);

  let out = "";
  let index = 0;
  while (index < text.length) {
    const consonant = consonantKeys.find((key) => text.startsWith(key, index));
    if (consonant) {
      let base = CONSONANTS[consonant];
      const doubled = consonant.length === 1 && text[index + 1] === consonant;
      const advance = doubled ? 2 : consonant.length;
      if (doubled) base = base + "്" + base;
      const sign = signKeys.find((key) => text.startsWith(key, index + advance));
      if (sign !== undefined) {
        out += base + SIGNS[sign];
        index += advance + sign.length;
        continue;
      }
      const isLast = index + advance >= text.length;
      if (isLast && !doubled && base === "മ") {
        out += "ം";
      } else if (isLast && !doubled && CHILLUS[base]) {
        out += CHILLUS[base];
      } else {
        out += base + "്";
      }
      index += advance;
      continue;
    }
    const vowel = vowelKeys.find((key) => text.startsWith(key, index));
    if (vowel) {
      out += VOWELS[vowel];
      index += vowel.length;
      continue;
    }
    out += text[index];
    index += 1;
  }
  return out;
}

async function loadModel() {
  const [encoder, decoder, tokenizerPayload] = await Promise.all([
    ort.InferenceSession.create("./models/encoder.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("./models/decoder.onnx", { executionProviders: ["wasm"] }),
    fetch("./models/tokenizer.json").then((response) => response.json()),
  ]);

  const tokenizer = buildTokenizer(tokenizerPayload);
  engine = new Transliterator({ encoder, decoder }, tokenizer, ort, {
    maxSrc: MAX_SRC,
    maxTgt: MAX_TGT,
    beam: BEAM,
  });

  elements.status.textContent = "model ready";
  elements.status.classList.remove("error");
}

async function loadBenchmarks() {
  try {
    const response = await fetch("./models/eval.json");
    if (!response.ok) throw new Error("missing");
    const data = await response.json();
    renderBenchmarks(data);
  } catch {
    elements.benchNote.textContent =
      "Benchmark table not built yet. Run scripts/run_baselines.py and model/evaluate.py to produce it.";
  }
}

function renderBenchmarks(data) {
  const rows = [];
  for (const entry of data) {
    const label = entry.file === "test_aksharantar_adhoc.tsv" ? "adhoc spelling" : "careful spelling";
    for (const [name, result] of Object.entries(entry.systems ?? {})) {
      rows.push({ file: label, name, ...result });
    }
  }
  if (!rows.length) return;

  elements.benchBody.innerHTML = "";
  let currentFile = null;
  for (const row of rows) {
    if (row.file !== currentFile) {
      currentFile = row.file;
      const header = document.createElement("tr");
      header.innerHTML = `<td colspan="5" style="padding-top:0.6rem;color:#6f6a75">${row.file}</td>`;
      elements.benchBody.appendChild(header);
    }
    const tr = document.createElement("tr");
    if (row.name.includes("our model")) tr.className = "ours";
    tr.innerHTML = `
      <td>${row.name}</td>
      <td class="num">${(row.top1 * 100).toFixed(1)}%</td>
      <td class="num">${(row.cer * 100).toFixed(1)}%</td>
      <td class="num">${((row.top5 ?? 0) * 100).toFixed(1)}%</td>
      <td class="num">${row.size ?? "-"}</td>
    `;
    elements.benchBody.appendChild(tr);
  }
}

async function update() {
  const text = normalizeRoman(elements.word.value);
  elements.ruleOutput.textContent = text ? ruleConvert(text) : " ";
  if (!engine || !text) {
    elements.modelOutput.textContent = " ";
    elements.candidates.innerHTML = "";
    return;
  }

  const current = ++requestId;
  const started = performance.now();
  const words = text.split(/\s+/).filter((word) => word.length);
  const last = words[words.length - 1];
  const translated = await engine.transliterate(text);
  if (current !== requestId) return;

  elements.modelOutput.textContent = translated;
  const candidates = await engine.candidates(last, 5);
  if (current !== requestId) return;
  elements.candidates.innerHTML = candidates.map((candidate) => `<li>${candidate}</li>`).join("");
  const elapsed = Math.round(performance.now() - started);
  elements.status.textContent = `model ready · ${elapsed} ms`;
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(update, 90);
}

elements.word.addEventListener("input", schedule);
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    elements.word.value = button.dataset.word;
    schedule();
  });
});

loadModel()
  .then(() => {
    loadBenchmarks();
    elements.word.focus();
  })
  .catch((error) => {
    elements.status.textContent = `could not load the model: ${error.message}`;
    elements.status.classList.add("error");
  });

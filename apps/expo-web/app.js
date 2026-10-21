import { Transliterator, buildTokenizer, normalizeRoman } from "./transliterator.js";
import { ruleConvert } from "./suggest.js";

const MAX_SRC = 48;
const MAX_TGT = 64;

const elements = {
  status: document.getElementById("status"),
  word: document.getElementById("word"),
  modelOutput: document.getElementById("model-output"),
  ruleOutput: document.getElementById("rule-output"),
  strip: document.getElementById("suggestion-strip"),
  benchBody: document.querySelector("#bench-table tbody"),
  benchNote: document.getElementById("bench-note"),
};

let worker = null;
let timer = null;
let requestId = 0;

function startWorker() {
  worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = (event) => {
    const { id, candidates, error, ready, threads, text, ranked } = event.data;
    if (error) {
      elements.status.textContent = `model error: ${error}`;
      elements.status.classList.add("error");
      return;
    }
    if (ready) {
      elements.status.textContent = `model ready · ${threads} threads`;
      elements.status.classList.remove("error");
    }
    if (id !== requestId) return;
    if (!candidates) return;
    elements.modelOutput.textContent = text ?? candidates[0] ?? " ";
    showStrip(ranked ?? candidates);
  };
}

function requestModel(text) {
  if (!worker) return;
  requestId += 1;
  worker.postMessage({ id: requestId, roman: text, nBest: 5 });
}

/** The suggestion strip, the way a keyboard shows it. */
function showStrip(words) {
  elements.strip.innerHTML = "";
  for (const word of words) {
    const button = document.createElement("button");
    button.className = "suggestion";
    button.textContent = word;
    button.addEventListener("click", () => applySuggestion(word));
    elements.strip.appendChild(button);
  }
}

/** Put the chosen word into the text, in place of the word being typed. */
function applySuggestion(word) {
  const value = elements.word.value.replace(/\S+\s*$/, "");
  elements.word.value = `${value}${word} `;
  elements.word.focus();
  update();
}

function update() {
  const text = normalizeRoman(elements.word.value);
  elements.ruleOutput.textContent = text ? ruleConvert(text) : " ";
  if (!text) {
    elements.modelOutput.textContent = " ";
    showStrip([]);
    return;
  }
  requestModel(text);
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(update, 80);
}

elements.word.addEventListener("input", schedule);
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    elements.word.value = button.dataset.word;
    schedule();
  });
});

async function loadBenchmarks() {
  try {
    const response = await fetch("./models/eval.json");
    if (!response.ok) throw new Error("missing");
    renderBenchmarks(await response.json());
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

startWorker();
loadBenchmarks();
elements.word.focus();

// Ask the worker to load the model straight away, so the first keystroke
// does not have to wait for it.
requestModel("");

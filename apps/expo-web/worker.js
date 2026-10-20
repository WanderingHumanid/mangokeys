/**
 * The model runs in a worker so that the page stays responsive while it
 * thinks. The rule engine on the main thread gives an instant preview and the
 * worker fills in the real candidates a moment later.
 */

import * as ort from "./ort/ort.min.mjs";
import { Transliterator, buildTokenizer } from "./transliterator.js";

const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = threads;
ort.env.wasm.proxy = false;

let engine = null;
let pending = null;
let running = false;

async function load() {
  const [encoder, decoder, tokenizerPayload] = await Promise.all([
    ort.InferenceSession.create("./models/encoder.onnx"),
    ort.InferenceSession.create("./models/decoder.onnx"),
    fetch("./models/tokenizer.json").then((response) => response.json()),
  ]);
  const tokenizer = buildTokenizer(tokenizerPayload);
  engine = new Transliterator({ encoder, decoder }, tokenizer, ort, {
    maxSrc: 48,
    maxTgt: 64,
    beam: 4,
  });
  return threads;
}

async function handle({ id, roman, nBest }) {
  if (!engine) {
    try {
      const count = await load();
      self.postMessage({ id, ready: true, threads: count });
    } catch (error) {
      self.postMessage({ id, error: String(error) });
      return;
    }
  }

  if (!roman) {
    self.postMessage({ id, text: "", candidates: [] });
    return;
  }

  const started = performance.now();
  try {
    const words = roman.split(/\s+/).filter((word) => word.length);
    const last = words[words.length - 1] ?? "";
    const candidates = last ? await engine.candidates(last, nBest) : [];
    const text = await engine.transliterate(roman);
    self.postMessage({ id, text, candidates, ms: performance.now() - started });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
}

async function drain() {
  running = true;
  while (pending) {
    const job = pending;
    pending = null;
    await handle(job);
  }
  running = false;
}

self.onmessage = (event) => {
  // Only the newest request matters. While one is running the others are
  // dropped, so fast typing does not build up a queue of stale work.
  pending = event.data;
  if (!running) drain();
};

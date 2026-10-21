/**
 * The model runs in a worker so that the page stays responsive while it
 * thinks. The rules and the word list run on the main thread, so the
 * suggestion strip is never empty, and the worker's answer refines it a
 * moment later.
 */

import * as ort from "./ort/ort.min.mjs";
import { Transliterator, buildTokenizer } from "./transliterator.js";

const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = threads;
ort.env.wasm.proxy = false;

let engine = null;
let prefixEngine = null;
let pending = null;
let running = false;

async function load() {
  const [encoder, decoder, prefixEncoder, prefixDecoder, tokenizerPayload] = await Promise.all([
    ort.InferenceSession.create("./models/encoder.onnx"),
    ort.InferenceSession.create("./models/decoder.onnx"),
    ort.InferenceSession.create("./models/prefix-encoder.onnx").catch(() => null),
    ort.InferenceSession.create("./models/prefix-decoder.onnx").catch(() => null),
    fetch("./models/tokenizer.json").then((response) => response.json()),
  ]);
  const tokenizer = buildTokenizer(tokenizerPayload);
  engine = new Transliterator({ encoder, decoder }, tokenizer, ort, {
    maxSrc: 48,
    maxTgt: 64,
    beam: 4,
  });
  if (prefixEncoder && prefixDecoder) {
    prefixEngine = new Transliterator({ encoder: prefixEncoder, decoder: prefixDecoder }, tokenizer, ort, {
      maxSrc: 48,
      maxTgt: 64,
      beam: 1,
    });
  }
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
    self.postMessage({ id, candidates: [], scores: [] });
    return;
  }

  const started = performance.now();
  try {
    const scored = await engine.scoredCandidates(roman, nBest);
    const candidates = scored.map((item) => item.text);
    const scores = scored.map((item) => item.score);

    // The prefix model stops in the middle of a word. The word list on the
    // main thread completes it, because the list is already loaded there.
    let prefix = "";
    if (prefixEngine) {
      prefix = await prefixEngine.transliterate(roman);
    }

    self.postMessage({
      id,
      candidates,
      scores,
      prefix,
      ms: performance.now() - started,
    });
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

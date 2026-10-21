/**
 * The model runs in a worker so that the page stays responsive while it
 * thinks. The rules and the word list run on the main thread, so the
 * suggestion strip is never empty, and the worker fills in the answer of the
 * model a moment later.
 */

import * as ort from "./ort/ort.min.mjs";
import { Transliterator, buildTokenizer } from "./transliterator.js";
import { Lexicon } from "./lexicon.js";
import { CharNgram } from "./ngram.js";
import { SuggestionEngine } from "./suggest.js";

const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = threads;
ort.env.wasm.proxy = false;

let engine = null;
let prefixEngine = null;
let ranker = null;
let pending = null;
let running = false;

async function load() {
  const [encoder, decoder, prefixEncoder, prefixDecoder, tokenizerPayload, lexicon, ngram] =
    await Promise.all([
      ort.InferenceSession.create("./models/encoder.onnx"),
      ort.InferenceSession.create("./models/decoder.onnx"),
      ort.InferenceSession.create("./models/prefix-encoder.onnx").catch(() => null),
      ort.InferenceSession.create("./models/prefix-decoder.onnx").catch(() => null),
      fetch("./models/tokenizer.json").then((response) => response.json()),
      Lexicon.load("./models/lexicon"),
      CharNgram.load("./models/ngram.txt").catch(() => null),
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
  ranker = new SuggestionEngine(lexicon, ngram);
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
    const scored = last ? await engine.scoredCandidates(last, nBest) : [];
    const candidates = scored.map((item) => item.text);
    const scores = scored.map((item) => item.score);

    // The prefix model stops in the middle of a word, and the word list
    // completes it. That is the "type four letters and get the word" part.
    let completion = [];
    if (last && prefixEngine) {
      const prefix = await prefixEngine.transliterate(last);
      completion = prefix ? ranker.lexicon.complete(prefix, 8) : [];
    }

    const ranked = ranker.suggest(last, candidates, scores, completion);
    const text = await engine.transliterate(roman);
    self.postMessage({
      id,
      text,
      candidates,
      ranked: ranked.map((item) => item.text),
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

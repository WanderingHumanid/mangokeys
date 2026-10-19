/**
 * Check the browser runtime against the exported model.
 *
 *   node runtime/js/test_runtime.mjs runs/ml_mini/onnx runs/ml_mini/tokenizer.json
 *
 * The same code runs in the browser through onnxruntime-web. Running it here
 * under onnxruntime-node catches the mistakes before they reach a demo.
 */

import * as ort from "onnxruntime-node";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { Transliterator, buildTokenizer } from "./transliterator.js";

const onnxDir = process.argv[2] ?? "runs/ml_mini/onnx";
const tokenizerPath = process.argv[3] ?? "runs/ml_mini/tokenizer.json";

const encoder = await ort.InferenceSession.create(join(onnxDir, "encoder.onnx"));
const decoder = await ort.InferenceSession.create(join(onnxDir, "decoder.onnx"));
const tokenizer = buildTokenizer(JSON.parse(readFileSync(tokenizerPath, "utf8")));

const engine = new Transliterator({ encoder, decoder }, tokenizer, ort, {
  maxSrc: 48,
  maxTgt: 64,
  beam: 4,
});

const words = [
  "ente",
  "peru",
  "kerala",
  "sundari",
  "njangalkum",
  "kaalam",
  "aavashyappettappol",
  "sughamano",
  "vakku",
  "ningalkkum",
];

const results = {};
let failures = 0;
for (const word of words) {
  const start = Date.now();
  const candidates = await engine.candidates(word, 5);
  const elapsed = Date.now() - start;
  results[word] = candidates;
  console.log(`${word.padEnd(22)} ${String(elapsed).padStart(5)} ms  ${candidates.join(" | ")}`);
  if (!candidates.length) {
    console.log("  no candidates");
    failures += 1;
  }
}

if (process.argv[4]) {
  writeFileSync(process.argv[4], JSON.stringify(results, null, 2));
  console.log(`wrote ${process.argv[4]}`);
}

if (failures) {
  console.error(`${failures} words produced nothing`);
  process.exit(1);
}
console.log("all words produced candidates");

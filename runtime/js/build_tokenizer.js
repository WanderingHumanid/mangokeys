/**
 * Build the tokenizer json that the browser runtime expects.
 *
 * The training run writes src and tgt vocabularies in one file. The browser
 * only needs the character to id mapping, so we flatten it here.
 *
 *   node runtime/js/build_tokenizer.js runs/ml_mini/tokenizer.json
 */

import { readFileSync, writeFileSync } from "node:fs";

const input = process.argv[2] ?? "runs/ml_mini/tokenizer.json";
const output = process.argv[3] ?? "runtime/js/tokenizer.json";

const payload = JSON.parse(readFileSync(input, "utf8"));
writeFileSync(output, JSON.stringify({ src: payload.src, tgt: payload.tgt }));
console.log(`wrote ${output}`);
console.log(`source vocabulary ${payload.src.length}, target vocabulary ${payload.tgt.length}`);

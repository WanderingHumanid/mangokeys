/**
 * Browser inference for the transliteration model.
 *
 * The two ONNX graphs are driven the same way as in the Python and Kotlin
 * runtimes: encode once, then feed the target prefix back in and collect the
 * logits. The decoder graph takes the causal mask as an input because a mask
 * built from a length that is only known at run time would be frozen at export
 * time.
 */

const MALAYALAM_BLOCK_START = 0x0d00;
const MALAYALAM_BLOCK_END = 0x0d80;

const LEGACY_CHILLU = {
  "ണ": "ൺ",
  "ന": "ൻ",
  "ര": "ർ",
  "ല": "ൽ",
  "ള": "ൾ",
  "ക": "ൿ",
};

const VIRAMA = "്";
const ANUSVARA = "ം";
const ZWJ = "‍";
const ZWNJ = "‌";

export function normalizeMalayalam(text) {
  let out = text.normalize("NFC");
  for (const [consonant, chillu] of Object.entries(LEGACY_CHILLU)) {
    out = out.split(consonant + VIRAMA + ANUSVARA).join(chillu);
    out = out.split(consonant + VIRAMA + ZWJ).join(chillu);
  }
  return out;
}

export function normalizeRoman(text) {
  return text
    .normalize("NFC")
    .replace(/[‘’‚‛]/g, "'")
    .replace(/[“”„«»]/g, '"')
    .replace(/[‐‑‒–—―−]/g, "-")
    .replace(/[‍⁠﻿]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function causalMask(length) {
  const mask = new Uint8Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = i + 1; j < length; j += 1) {
      mask[i * length + j] = 1;
    }
  }
  return mask;
}

function logSoftmax(values) {
  let max = -Infinity;
  for (const value of values) {
    if (value > max) max = value;
  }
  let sum = 0;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = values[i] - max;
    sum += Math.exp(out[i]);
  }
  const logSum = Math.log(sum);
  for (let i = 0; i < out.length; i += 1) {
    out[i] -= logSum;
  }
  return out;
}

export class Transliterator {
  /**
   * @param {object} session encoder and decoder sessions
   * @param {object} tokenizer with src and target vocabularies
   * @param {object} options maxSrc, maxTgt and beam size
   */
  constructor(session, tokenizer, options = {}) {
    this.encoder = session.encoder;
    this.decoder = session.decoder;
    this.tokenizer = tokenizer;
    this.maxSrc = options.maxSrc ?? 48;
    this.maxTgt = options.maxTgt ?? 64;
    this.beam = options.beam ?? 4;
  }

  encodeSource(roman) {
    const ids = [];
    for (const ch of roman) {
      const id = this.tokenizer.src.get(ch);
      ids.push(id === undefined ? this.tokenizer.unk : id);
      if (ids.length >= this.maxSrc) break;
    }
    return ids;
  }

  decodeTarget(ids) {
    let out = "";
    for (const id of ids) {
      const ch = this.tokenizer.tgt.itos[id];
      if (ch !== undefined && !ch.startsWith("<")) out += ch;
    }
    return out;
  }

  async candidates(roman, nBest = 5) {
    const clean = normalizeRoman(roman);
    if (!clean) return [];

    const srcIds = this.encodeSource(clean);
    const src = new BigInt64Array(this.maxSrc);
    const srcMask = new Uint8Array(this.maxSrc);
    for (let i = 0; i < srcIds.length; i += 1) {
      src[i] = BigInt(srcIds[i]);
      srcMask[i] = 1;
    }

    const memory = await this.encoder.run({ src_ids: src, src_mask: srcMask });
    const width = Math.min(this.beam, this.maxTgt);
    const beamMemory = new BigInt64Array(memory.dims[0] * width * memory.dims[1] * memory.dims[2]);
    for (let b = 0; b < width; b += 1) {
      const copy = memory.data.slice(b * memory.dims[1] * memory.dims[2], (b + 1) * memory.dims[1] * memory.dims[2]);
      beamMemory.set(copy, b * memory.dims[1] * memory.dims[2]);
    }
    const beamMask = new Uint8Array(this.maxSrc * width);
    for (let b = 0; b < width; b += 1) {
      beamMask.set(srcMask, b * this.maxSrc);
    }

    let tokens = new BigInt64Array(width);
    for (let i = 0; i < width; i += 1) tokens[i] = BigInt(this.tokenizer.bos);

    let scores = new Float32Array(width).fill(-Infinity);
    scores[0] = 0;
    const finished = [];

    for (let step = 0; step < this.maxTgt; step += 1) {
      const logits = await this.decoder.run({
        tgt_prefix: tokens,
        memory: beamMemory,
        src_mask: beamMask,
        self_mask: causalMask(tokens.length),
      });

      const vocab = logits.dims[1];
      const candidates = new Array(width * vocab);
      for (let b = 0; b < width; b += 1) {
        const row = logSoftmax(logits.data.subarray(b * vocab, (b + 1) * vocab));
        for (let v = 0; v < vocab; v += 1) {
          candidates[b * vocab + v] = scores[b] + row[v];
        }
      }

      const order = candidates
        .map((score, index) => [score, index])
        .sort((a, b) => b[0] - a[0])
        .slice(0, width);

      const nextTokens = new BigInt64Array(width);
      const nextScores = new Float32Array(width);
      const nextPrefix = new BigInt64Array(width * (tokens.length + 1));
      for (let slot = 0; slot < width; slot += 1) {
        const [, index] = order[slot];
        const parent = Math.floor(index / vocab);
        const token = index % vocab;
        nextPrefix.set(tokens.subarray(parent * tokens.length, (parent + 1) * tokens.length), slot * tokens.length);
        nextPrefix[slot * (tokens.length + 1) + tokens.length] = BigInt(token);
        nextTokens[slot] = BigInt(token);
        nextScores[slot] = order[slot][0];

        if (token === this.tokenizer.eos) {
          const sequence = Array.from(nextPrefix.subarray(slot * (tokens.length + 1), slot * (tokens.length + 1) + tokens.length));
          finished.push({ score: order[slot][0] / Math.max(sequence.length, 1), sequence });
        }
      }

      tokens = nextPrefix;
      scores = nextScores;
      let alive = false;
      for (let slot = 0; slot < width; slot += 1) {
        if (nextTokens[slot] === BigInt(this.tokenizer.eos)) {
          scores[slot] = -Infinity;
        } else if (scores[slot] > -Infinity) {
          alive = true;
        }
      }
      if (!alive) break;
    }

    if (finished.length === 0) {
      for (let slot = 0; slot < width; slot += 1) {
        if (scores[slot] === -Infinity) continue;
        const sequence = Array.from(tokens.subarray(slot * tokens.length, (slot + 1) * tokens.length));
        finished.push({ score: scores[slot] / Math.max(sequence.length, 1), sequence });
      }
    }

    finished.sort((a, b) => b.score - a.score);
    return finished.slice(0, nBest).map((item) => normalizeMalayalam(this.decodeTarget(item.sequence)));
  }

  async transliterate(roman) {
    const result = await this.candidates(roman, 1);
    return result.length ? result[0] : "";
  }
}

/** Build a tokenizer from the json file written next to the model. */
export function buildTokenizer(payload) {
  const src = new Map();
  const tgt = new Map();
  payload.src.forEach((ch, index) => src.set(ch, index));
  payload.tgt.forEach((ch, index) => tgt.set(ch, index));
  return {
    src,
    tgt: { itos: payload.tgt },
    bos: payload.tgt.indexOf("<bos>"),
    eos: payload.tgt.indexOf("<eos>"),
    unk: payload.tgt.indexOf("<unk>"),
  };
}

export { MALAYALAM_BLOCK_START, MALAYALAM_BLOCK_END };

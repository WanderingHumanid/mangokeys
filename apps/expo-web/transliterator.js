/**
 * Browser inference for the transliteration model.
 *
 * The two ONNX graphs are driven the same way as in the Python and Kotlin
 * runtimes: encode once, then feed the target prefix back in and collect the
 * logits. The decoder graph takes the causal mask as an input because a mask
 * built from a length that is only known at run time would be frozen at export
 * time.
 */

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
  for (let row = 0; row < length; row += 1) {
    for (let column = row + 1; column < length; column += 1) {
      mask[row * length + column] = 1;
    }
  }
  return mask;
}

function logSoftmax(values, offset, size) {
  let max = -Infinity;
  for (let i = 0; i < size; i += 1) {
    if (values[offset + i] > max) max = values[offset + i];
  }
  let sum = 0;
  const out = new Float64Array(size);
  for (let i = 0; i < size; i += 1) {
    out[i] = values[offset + i] - max;
    sum += Math.exp(out[i]);
  }
  const logSum = Math.log(sum);
  for (let i = 0; i < size; i += 1) out[i] -= logSum;
  return out;
}

export class Transliterator {
  /**
   * @param {object} sessions encoder and decoder ONNX sessions
   * @param {object} tokenizer src and target vocabularies
   * @param {object} ort the onnxruntime module
   */
  constructor(sessions, tokenizer, ort, options = {}) {
    this.encoder = sessions.encoder;
    this.decoder = sessions.decoder;
    this.tokenizer = tokenizer;
    this.ort = ort;
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

    const ort = this.ort;
    const srcIds = this.encodeSource(clean);
    const srcData = new BigInt64Array(this.maxSrc);
    const maskData = new Uint8Array(this.maxSrc);
    for (let i = 0; i < srcIds.length; i += 1) {
      srcData[i] = BigInt(srcIds[i]);
      maskData[i] = 1;
    }

    const encoded = await this.encoder.run({
      src_ids: new ort.Tensor("int64", srcData, [1, this.maxSrc]),
      src_mask: new ort.Tensor("bool", maskData, [1, this.maxSrc]),
    });
    const memory = encoded.memory;

    const width = Math.min(this.beam, this.maxTgt);
    const rowSize = memory.dims[1] * memory.dims[2];
    const beamData = new Float32Array(rowSize * width);
    for (let slot = 0; slot < width; slot += 1) {
      // The encoder produced one row. Every beam slot works on the same one.
      beamData.set(memory.data.subarray(0, rowSize), slot * rowSize);
    }
    const beamMaskData = new Uint8Array(this.maxSrc * width);
    for (let slot = 0; slot < width; slot += 1) {
      beamMaskData.set(maskData, slot * this.maxSrc);
    }

    const beamMemory = new ort.Tensor("float32", beamData, [width, memory.dims[1], memory.dims[2]]);
    const beamMask = new ort.Tensor("bool", beamMaskData, [width, this.maxSrc]);

    let prefixLength = 1;
    let tokens = new BigInt64Array(width);
    for (let i = 0; i < width; i += 1) tokens[i] = BigInt(this.tokenizer.bos);

    let scores = new Float64Array(width).fill(-Infinity);
    scores[0] = 0;
    const finished = [];

    for (let step = 0; step < this.maxTgt; step += 1) {
      const outputs = await this.decoder.run({
        tgt_prefix: new ort.Tensor("int64", tokens, [width, prefixLength]),
        memory: beamMemory,
        src_mask: beamMask,
        self_mask: new ort.Tensor("bool", causalMask(prefixLength), [1, 1, prefixLength, prefixLength]),
      });

      const logits = outputs.logits;
      const vocab = logits.dims[1];
      const ranked = [];
      for (let slot = 0; slot < width; slot += 1) {
        const row = logSoftmax(logits.data, slot * vocab, vocab);
        for (let token = 0; token < vocab; token += 1) {
          ranked.push({ score: scores[slot] + row[token], parent: slot, token });
        }
      }
      ranked.sort((a, b) => b.score - a.score);

      const nextLength = prefixLength + 1;
      const nextTokens = new BigInt64Array(width);
      const nextScores = new Float64Array(width);
      const nextPrefix = new BigInt64Array(width * nextLength);
      for (let slot = 0; slot < width; slot += 1) {
        const { score, parent, token } = ranked[slot];
        const prefixStart = slot * nextLength;
        nextPrefix.set(
          tokens.subarray(parent * prefixLength, (parent + 1) * prefixLength),
          prefixStart,
        );
        nextPrefix[prefixStart + prefixLength] = BigInt(token);
        nextTokens[slot] = BigInt(token);
        nextScores[slot] = score;

        if (token === this.tokenizer.eos) {
          const sequence = Array.from(
            nextPrefix.subarray(prefixStart, prefixStart + prefixLength),
            (value) => Number(value),
          );
          finished.push({ score: score / Math.max(sequence.length, 1), sequence });
        }
      }

      tokens = nextPrefix;
      prefixLength = nextLength;
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
        const sequence = Array.from(
          tokens.subarray(slot * prefixLength, (slot + 1) * prefixLength),
          (value) => Number(value),
        );
        finished.push({ score: scores[slot] / Math.max(sequence.length, 1), sequence });
      }
    }

    finished.sort((a, b) => b.score - a.score);
    return finished
      .slice(0, nBest)
      .map((item) => normalizeMalayalam(this.decodeTarget(item.sequence)));
  }

  async transliterate(roman) {
    const words = normalizeRoman(roman).split(/\s+/).filter((word) => word.length);
    if (!words.length) return "";
    if (words.length === 1) {
      const result = await this.candidates(words[0], 1);
      return result.length ? result[0] : "";
    }
    const parts = [];
    for (const word of words) {
      const result = await this.candidates(word, 1);
      parts.push(result.length ? result[0] : "");
    }
    return parts.join(" ");
  }
}

/** Build a tokenizer from the json file written next to the model. */
export function buildTokenizer(payload) {
  const src = new Map();
  payload.src.forEach((ch, index) => src.set(ch, index));
  return {
    src,
    tgt: { itos: payload.tgt },
    bos: payload.tgt.indexOf("<bos>"),
    eos: payload.tgt.indexOf("<eos>"),
    unk: payload.tgt.indexOf("<unk>"),
  };
}

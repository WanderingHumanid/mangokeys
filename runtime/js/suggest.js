/**
 * The ranker of the keyboard in the browser.
 *
 * The rules and the model both propose words, the word list turns them into
 * real words, and the language model and the corpus frequency decide which
 * one comes first. The same code runs on the phone.
 */

const CONSONANTS = {
  k: "ക", kh: "ഖ", g: "ഗ", gh: "ഘ", ng: "ങ", ch: "ച", chh: "ഛ",
  j: "ജ", jh: "ഝ", nj: "ഞ", t: "ട", T: "ട", th: "ഠ", D: "ഡ",
  dh: "ഢ", N: "ണ", thh: "ത", n: "ന", p: "പ", ph: "ഫ", b: "ബ",
  bh: "ഭ", m: "മ", y: "യ", r: "ര", l: "ല", v: "വ", sh: "ശ",
  Sh: "ഷ", s: "സ", h: "ഹ", L: "ള", zh: "ഴ", R: "റ", rr: "റ",
};
const VOWELS = { a: "അ", aa: "ആ", i: "ഇ", ii: "ഈ", u: "ഉ", uu: "ഊ", e: "എ", ee: "ഏ", ai: "ഐ", o: "ഒ", O: "ഓ", au: "ഔ" };
const SIGNS = { a: "", aa: "ാ", i: "ി", ii: "ീ", u: "ു", uu: "ൂ", e: "െ", ee: "േ", ai: "ൈ", o: "ൊ", oo: "ോ", au: "ൗ" };
const CHILLUS = { "ണ": "ൺ", "ന": "ൻ", "ര": "ർ", "ല": "ൽ", "ള": "ൾ", "ക": "ൿ" };

/** The rule based converter, the instant proposer. */
export function ruleConvert(text) {
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

export const DEFAULT_WEIGHTS = {
  model: 1.0,
  language: 1.0,
  frequency: 0.2,
  unknown: -2.0,
  rulePenalty: -12.0,
  limit: 8,
};

export class SuggestionEngine {
  constructor(lexicon, ngram, weights = {}) {
    this.lexicon = lexicon;
    this.ngram = ngram;
    this.weights = { ...DEFAULT_WEIGHTS, ...weights };
  }

  suggest(roman, modelCandidates = [], modelScores = [], useRules = true) {
    const text = roman.trim();
    if (!text) return [];

    const pool = new Map();
    const offer = (word, source, score) => {
      const known = pool.get(word);
      if (known) {
        known.modelScore = Math.max(known.modelScore, score);
        return;
      }
      pool.set(word, { text: word, source, modelScore: score, lmScore: 0, count: 0, total: 0 });
      for (const neighbour of this.lexicon.expand(word, 4)) {
        if (neighbour === word) continue;
        const other = pool.get(neighbour);
        if (other) other.modelScore = Math.max(other.modelScore, score);
        else pool.set(neighbour, { text: neighbour, source, modelScore: score, lmScore: 0, count: 0, total: 0 });
      }
    };

    modelCandidates.forEach((word, index) => {
      if (!word) return;
      offer(word, "model", modelScores[index] ?? 0);
    });

    if (useRules) {
      for (const word of ruleConvert(text).slice(0, 1)) offer(word, "rules", this.weights.rulePenalty);
    }

    for (const candidate of pool.values()) {
      candidate.count = this.lexicon.count(candidate.text);
      candidate.lmScore = this.ngram ? this.ngram.meanScore(candidate.text) : 0;
      candidate.total =
        this.weights.model * candidate.modelScore +
        this.weights.language * candidate.lmScore +
        this.weights.frequency * Math.log1p(candidate.count);
      if (!candidate.count) candidate.total += this.weights.unknown;
    }

    return [...pool.values()]
      .sort((a, b) => b.total - a.total)
      .slice(0, this.weights.limit);
  }
}

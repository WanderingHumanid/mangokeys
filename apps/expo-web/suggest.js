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
  known: 4.0,
  completion: 0.0,
  variation: -2.0,
  language: 0.0,
  frequency: 0.0,
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
    const best = modelScores.length ? modelScores[0] : 0;

    const add = (word, source, score) => {
      const known = pool.get(word);
      if (known) {
        known.modelScore = Math.max(known.modelScore, score);
        return known;
      }
      const candidate = {
        text: word,
        source,
        modelScore: score,
        lmScore: 0,
        count: 0,
        known: false,
        completion: false,
        variation: false,
        total: 0,
      };
      pool.set(word, candidate);
      return candidate;
    };

    const offer = (word, source, score) => {
      const candidate = add(word, source, score);
      // The answer is a real word when the word list knows it. The words
      // that the model itself proposed get the bonus, because the model is
      // the better judge of what the user meant.
      if (source === "model" && this.lexicon.contains(word)) candidate.known = true;

      // The words that start with the answer are the completion. They are
      // the answer only when the model did not give a whole word.
      if (!candidate.known) {
        for (const other of this.lexicon.complete(word, 3)) {
          if (other === word) continue;
          const neighbour = add(other, source, score);
          neighbour.completion = true;
        }
      }

      // The words that resemble the answer are the variations.
      for (const item of this.lexicon.nearestWithDistance(word, 3)) {
        if (item.word === word) continue;
        const neighbour = add(item.word, source, score);
        neighbour.variation = true;
      }
    };

    modelCandidates.forEach((word, index) => {
      if (!word) return;
      offer(word, "model", (modelScores[index] ?? best) - best);
    });

    if (useRules) {
      for (const word of ruleConvert(text).slice(0, 1)) offer(word, "rules", this.weights.rulePenalty);
    }

    for (const candidate of pool.values()) {
      candidate.count = this.lexicon.count(candidate.text);
      candidate.lmScore = this.ngram ? this.ngram.meanScore(candidate.text) : 0;
      candidate.total =
        this.weights.model * candidate.modelScore +
        this.weights.known * (candidate.known ? 1 : 0) +
        this.weights.completion * (candidate.completion ? 1 : 0) +
        this.weights.variation * (candidate.variation ? 1 : 0) +
        this.weights.language * candidate.lmScore +
        this.weights.frequency * Math.log1p(candidate.count);
    }

    return [...pool.values()]
      .sort((a, b) => b.total - a.total)
      .slice(0, this.weights.limit);
  }
}

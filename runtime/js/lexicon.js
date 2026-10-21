/**
 * The Malayalam word list in the browser.
 *
 * The same three lookups as the python side: the word closest to a string,
 * the words that start with a prefix, and the combination of the two that
 * the keyboard uses. The files are the ones that scripts/build_lexicon.py
 * writes, so the browser and the phone use the same word list.
 */

const MARKS = new Set("ാിീുൂൃൄ൅െേൈൊോൌൗ്ം");
const CHILLU_BASE = {
  "ൺ": "ണ",
  "ൻ": "ന",
  "ർ": "ര",
  "ൽ": "ല",
  "ൾ": "ള",
  "ൿ": "ക",
};
const SEPARATOR = "";
const ENTRY_SIZE = 12;

/** The same 64 bit hash the python side uses to build the index. */
export function fnv1a(text) {
  let value = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  const bytes = new TextEncoder().encode(text);
  for (const byte of bytes) {
    value = (value ^ BigInt(byte)) & mask;
    value = (value * prime) & mask;
  }
  return value.toString();
}

/** Split text into Malayalam letters, the same way the python side does. */
export function letters(text) {
  const out = [];
  for (const char of text) {
    const code = char.codePointAt(0);
    const hanging =
      (code >= 0x0d3b && code <= 0x0d4d) ||
      code === 0x0d01 ||
      code === 0x0d02 ||
      code === 0x0d03 ||
      code === 0x0d55 ||
      code === 0x0d56 ||
      code === 0x0d57 ||
      code === 0x200c ||
      code === 0x200d;
    if (hanging && out.length) out[out.length - 1] += char;
    else out.push(char);
  }
  return out;
}

/** The consonant letters of a word, without the marks and the chillus. */
export function skeleton(word) {
  let out = "";
  for (const char of word) {
    const letter = CHILLU_BASE[char] ?? char;
    if ([...letter].length !== 1) continue;
    if (MARKS.has(letter)) continue;
    const code = letter.codePointAt(0);
    if (code >= 0x0d15 && code <= 0x0d39) out += letter;
  }
  return out;
}

/** The edit distance between two lists of letters. */
export function levenshtein(a, b) {
  if (a.length < b.length) [a, b] = [b, a];
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const current = [i];
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current.push(Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost));
    }
    previous = current;
  }
  return previous[b.length];
}

/** Every string that comes from deleting up to `distance` items. */
function deletions(sequence, distance) {
  const found = new Set([sequence.join(SEPARATOR)]);
  let frontier = [sequence];
  for (let step = 0; step < distance; step += 1) {
    const next = [];
    for (const item of frontier) {
      for (let index = 0; index < item.length; index += 1) {
        const shorter = item.slice(0, index).concat(item.slice(index + 1));
        next.push(shorter);
        found.add(shorter.join(SEPARATOR));
      }
    }
    frontier = next;
  }
  return found;
}

export class Lexicon {
  constructor(words, counts, index, maxDistance = 2) {
    this.words = words;
    this.counts = counts;
    this.maxDistance = maxDistance;
    this.wordIds = new Map(words.map((word, index) => [word, index]));
    this.index = index;
    this.skeletons = words.map(skeleton);
    this.buckets = { 1: new Map(), 2: new Map(), 3: new Map() };
    for (let id = 0; id < words.length; id += 1) {
      const key = this.skeletons[id];
      for (const depth of [1, 2, 3]) {
        if (key.length < depth) continue;
        const name = key.slice(0, depth);
        if (!this.buckets[depth].has(name)) this.buckets[depth].set(name, []);
        this.buckets[depth].get(name).push(id);
      }
    }
    for (const depth of [1, 2, 3]) {
      for (const bucket of this.buckets[depth].values()) {
        bucket.sort((a, b) => this.counts[b] - this.counts[a]);
      }
    }
  }

  static async load(directory, maxDistance = 2) {
    const [wordText, indexBuffer] = await Promise.all([
      fetch(`${directory}/words.txt`).then((response) => response.text()),
      fetch(`${directory}/index.bin`).then((response) => response.arrayBuffer()),
    ]);
    const words = [];
    const counts = [];
    for (const line of wordText.split("\n")) {
      if (!line) continue;
      const parts = line.split("\t");
      words.push(parts[0]);
      counts.push(parts.length > 1 ? Number(parts[1]) : 0);
    }
    const view = new DataView(indexBuffer);
    const size = view.getUint32(4, true);
    const index = new Map();
    for (let position = 0; position < size; position += 1) {
      const offset = 8 + position * ENTRY_SIZE;
      const key = view.getBigUint64(offset, true);
      const id = view.getUint32(offset + 8, true);
      const name = key.toString();
      if (!index.has(name)) index.set(name, []);
      index.get(name).push(id);
    }
    return new Lexicon(words, counts, index, maxDistance);
  }

  contains(word) {
    return this.wordIds.has(word);
  }

  count(word) {
    const id = this.wordIds.get(word);
    return id === undefined ? 0 : this.counts[id];
  }

  nearest(word, limit = 8) {
    if (!word) return [];
    const items = letters(word);
    const found = new Map();
    for (const variant of deletions(items, this.maxDistance)) {
      const ids = this.index.get(fnv1a(variant));
      if (!ids) continue;
      for (const id of ids) {
        const distance = levenshtein(items, letters(this.words[id]));
        if (distance > this.maxDistance) continue;
        const known = found.get(id);
        if (known === undefined || distance < known) found.set(id, distance);
      }
    }
    return [...found.entries()]
      .sort((a, b) => a[1] - b[1] || this.counts[b[0]] - this.counts[a[0]] || this.words[a[0]].localeCompare(this.words[b[0]]))
      .slice(0, limit)
      .map(([id]) => this.words[id]);
  }

  complete(prefix, limit = 8) {
    if (!prefix) return [];
    const key = skeleton(prefix);
    if (!key) return [];
    const depth = Math.min(key.length, 3);
    const bucket = this.buckets[depth].get(key.slice(0, depth)) ?? [];
    const out = [];
    for (const id of bucket) {
      if (!this.skeletons[id].startsWith(key)) continue;
      out.push(this.words[id]);
      if (out.length >= limit) break;
    }
    return out;
  }

  expand(text, limit = 8) {
    if (!text) return [];
    const out = [];
    if (this.contains(text)) out.push(text);
    for (const word of this.complete(text, limit * 2)) {
      if (!out.includes(word)) out.push(word);
      if (out.length >= limit) break;
    }
    if (out.length < limit) {
      for (const word of this.nearest(text, limit * 2)) {
        if (!out.includes(word)) out.push(word);
        if (out.length >= limit) break;
      }
    }
    return out.slice(0, limit);
  }
}

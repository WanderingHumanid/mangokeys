/**
 * The Malayalam character language model in the browser.
 *
 * It is trained on the same corpus as the transliterator and it ranks the
 * words that the word list and the model produce. The file is the one that
 * scripts/train_ngram.py writes.
 */

const START = "<s>";
const END = "</s>";
const SEPARATOR = "";

function letters(text) {
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

export class CharNgram {
  constructor(order, alpha, tables, vocabulary) {
    this.order = order;
    this.alpha = alpha;
    this.tables = tables;
    this.totals = tables.map((table) => {
      const totals = new Map();
      for (const [context, counts] of table) {
        let sum = 0;
        for (const value of counts.values()) sum += value;
        totals.set(context, sum);
      }
      return totals;
    });
    this.vocabulary = vocabulary;
  }

  static async load(path) {
    const text = await fetch(path).then((response) => response.text());
    let order = 3;
    let alpha = 0.1;
    const rows = [];
    const symbols = new Set();
    for (const line of text.split("\n")) {
      if (!line) continue;
      if (line.startsWith("order\t")) {
        order = Number(line.split("\t")[1]);
        continue;
      }
      if (line.startsWith("alpha\t")) {
        alpha = Number(line.split("\t")[1]);
        continue;
      }
      if (!line.startsWith("ngram\t")) continue;
      const parts = line.split("\t");
      const length = Number(parts[1]);
      const context = parts.slice(2, -2).join(SEPARATOR);
      const symbol = parts[parts.length - 2];
      const count = Number(parts[parts.length - 1]);
      rows.push([length, context, symbol, count]);
      symbols.add(symbol);
    }
    const tables = [];
    for (let index = 0; index < order; index += 1) tables.push(new Map());
    for (const [length, context, symbol, count] of rows) {
      const table = tables[length - 1];
      if (!table.has(context)) table.set(context, new Map());
      const counts = table.get(context);
      counts.set(symbol, (counts.get(symbol) ?? 0) + count);
      symbols.add(symbol);
    }
    return new CharNgram(order, alpha, tables, Math.max(symbols.size, 1));
  }

  probability(context, symbol) {
    for (let length = Math.min(context.length, this.order); length > 0; length -= 1) {
      const key = context.slice(context.length - length).join(SEPARATOR);
      const table = this.tables[length - 1].get(key);
      if (!table) continue;
      const total = this.totals[length - 1].get(key) ?? 0;
      if (!total) continue;
      return ((table.get(symbol) ?? 0) + this.alpha) / (total + this.alpha * this.vocabulary);
    }
    const table = this.tables[0].get("");
    const total = this.totals[0].get("") ?? 1;
    return ((table?.get(symbol) ?? 0) + this.alpha) / (total + this.alpha * this.vocabulary);
  }

  score(text) {
    const symbols = [START, ...letters(text), END];
    let total = 0;
    for (let index = 0; index < symbols.length; index += 1) {
      total += Math.log(this.probability(symbols.slice(0, index), symbols[index]));
    }
    return total;
  }

  meanScore(text) {
    if (!text) return 0;
    return this.score(text) / (letters(text).length + 2);
  }
}

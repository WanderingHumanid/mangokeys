"""A character language model over Malayalam.

The model in this project says which word the user probably meant. The
language model here says which of the words around that answer is the most
natural one. It is a small statistical model, trained on the same corpus as
the transliterator, and it does two jobs:

- it ranks the candidates, so that the common word comes before the rare one
- it scores a partial word, so that a prefix can be ranked too

It works on letters rather than on words, because a keyboard has to score
half typed words as well.
"""

from __future__ import annotations

import math
from pathlib import Path

from model.src.metrics import graphemes

START = "<s>"
END = "</s>"
SEPARATOR = "\x1f"
HEADER = "order\t{order}\nalpha\t{alpha}\n"


class CharNgram:
    """A character n-gram language model with stupid backoff."""

    def __init__(self, order: int = 3, alpha: float = 0.1) -> None:
        self.order = order
        self.alpha = alpha
        # tables[0] is the unigram table, keyed by the empty context.
        self.tables: list[dict[str, dict[str, int]]] = [{} for _ in range(order)]
        self.totals: list[dict[str, int]] = [{} for _ in range(order)]
        self.vocabulary = 1

    # --------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        words,
        order: int = 3,
        alpha: float = 0.1,
        min_freq: int = 1,
    ) -> "CharNgram":
        """Train on an iterable of native words.

        The n-grams that were seen only once are dropped when `min_freq` is
        more than one. They cost a lot of space and they say almost nothing
        about a word.
        """
        model = cls(order=order, alpha=alpha)
        symbols_seen: set[str] = set()
        for word in words:
            symbols = [START, *graphemes(word), END]
            symbols_seen.update(symbols)
            for index in range(len(symbols)):
                symbol = symbols[index]
                for length in range(1, order + 1):
                    if index - length + 1 < 0:
                        break
                    context = SEPARATOR.join(symbols[index - length + 1 : index])
                    table = model.tables[length - 1].setdefault(context, {})
                    table[symbol] = table.get(symbol, 0) + 1
        model.vocabulary = max(len(symbols_seen), 1)
        if min_freq > 1:
            for position, table in enumerate(model.tables):
                model.tables[position] = {
                    context: counts
                    for context, counts in table.items()
                    if sum(counts.values()) >= min_freq
                }
        model._finalize()
        return model

    def _finalize(self) -> None:
        for position, table in enumerate(self.tables):
            totals: dict[str, int] = {}
            for context, counts in table.items():
                totals[context] = sum(counts.values())
            self.totals[position] = totals

    # --------------------------------------------------------------- scoring

    def probability(self, context: list[str], symbol: str) -> float:
        """The probability of a symbol given the letters before it."""
        for length in range(min(len(context), self.order), 0, -1):
            key = SEPARATOR.join(context[len(context) - length :])
            table = self.tables[length - 1].get(key)
            if not table:
                continue
            total = self.totals[length - 1].get(key, 0)
            if total == 0:
                continue
            count = table.get(symbol, 0)
            return (count + self.alpha) / (total + self.alpha * self.vocabulary)

        table = self.tables[0].get("")
        total = self.totals[0].get("", 1)
        count = table.get(symbol, 0) if table else 0
        return (count + self.alpha) / (total + self.alpha * self.vocabulary)

    def score(self, text: str) -> float:
        """The log probability of a word. Longer words score lower."""
        symbols = [START, *graphemes(text), END]
        total = 0.0
        for index, symbol in enumerate(symbols):
            total += math.log(self.probability(symbols[:index], symbol))
        return total

    def mean_score(self, text: str) -> float:
        """The log probability per letter, so that words of different lengths
        can be compared."""
        if not text:
            return 0.0
        return self.score(text) / (len(graphemes(text)) + 2)

    def __len__(self) -> int:
        return sum(len(table) for table in self.tables)

    # ----------------------------------------------------------------- storage

    def save(self, path: str | Path) -> None:
        """Write the counts to a text file."""
        lines = [HEADER.format(order=self.order, alpha=self.alpha)]
        for position, table in enumerate(self.tables):
            length = position + 1
            for context, counts in table.items():
                context_parts = context.split(SEPARATOR) if context else []
                for symbol, count in counts.items():
                    row = [str(length), *context_parts, symbol, str(count)]
                    lines.append("ngram\t" + "\t".join(row) + "\n")
        Path(path).write_text("".join(lines), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharNgram":
        """Read a model back from a text file."""
        model = cls()
        order = 3
        alpha = 0.1
        symbols: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("order\t"):
                    order = int(line.split("\t")[1])
                elif line.startswith("alpha\t"):
                    alpha = float(line.split("\t")[1])
                elif line.startswith("ngram\t"):
                    parts = line.rstrip("\n").split("\t")
                    length = int(parts[1])
                    symbol = parts[-2]
                    count = int(parts[-1])
                    context = parts[2:-2]
                    symbols.add(symbol)
                    symbols.update(context)
        model = cls(order=order, alpha=alpha)
        model.vocabulary = max(len(symbols), 1)
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("ngram\t"):
                    continue
                parts = line.rstrip("\n").split("\t")
                length = int(parts[1])
                context = SEPARATOR.join(parts[2:-2])
                symbol = parts[-2]
                count = int(parts[-1])
                table = model.tables[length - 1].setdefault(context, {})
                table[symbol] = table.get(symbol, 0) + count
        model._finalize()
        return model

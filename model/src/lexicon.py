"""A Malayalam word list with the lookups a keyboard needs.

The error analysis says the model finds the right word and then misspells it.
Almost nine out of ten of its mistakes are one or two edits away from the
correct word, and two thirds of them are a missing or an extra vowel sign or
virama. That means the word is already there, and a word list is enough to
find it.

Three lookups are provided.

`nearest` finds the real word closest to a string. It uses a deletion index
in the spirit of SymSpell, so a lookup costs a few microseconds instead of a
scan over the whole list.

`complete` finds the real words that start with a given native string. The
lookup goes through the consonant skeleton of the word, so a prefix that has
the wrong vowel sign still finds its words.

`expand` is the combination the keyboard uses. It takes what the model said
and returns the real words around it, so the suggestion strip only ever shows
words that exist.
"""

from __future__ import annotations

import struct
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from model.src.metrics import graphemes, levenshtein

WORD_FILE = "words.txt"
INDEX_FILE = "index.bin"

# The marks that sit after a letter and do not make a letter of their own.
MARKS = set("ാിീുൂൃൄ൅െേൈൊോൌൗ്ം")

# The old chillu letters and the letters they stand for.
CHILLU_BASE = {
    "ൺ": "ണ",
    "ൻ": "ന",
    "ർ": "ര",
    "ൽ": "ല",
    "ൾ": "ള",
    "ൿ": "ക",
}

SEPARATOR = "\x1f"
ENTRY = struct.Struct("<QI")
HEADER_SIZE = 8

# The Malayalam consonants, from ക to ഹ.
CONSONANT_FIRST = 0x0D15
CONSONANT_LAST = 0x0D39


def fnv1a(text: str) -> int:
    """A stable 64 bit hash. The same function is used in the phone runtime."""
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def skeleton(word: str) -> str:
    """The consonant letters of a word, without the marks and the chillus.

    Two spellings of the same word differ in the vowel signs, and the
    consonant letters are what stays the same. A lookup on the skeleton
    therefore finds a word even when the vowel signs are wrong.
    """
    out: list[str] = []
    for char in word:
        letter = CHILLU_BASE.get(char, char)
        if len(letter) != 1 or letter in MARKS:
            continue
        code = ord(letter)
        if CONSONANT_FIRST <= code <= CONSONANT_LAST:
            out.append(letter)
    return "".join(out)


def _deletions(sequence: list[str], distance: int) -> set[str]:
    """Every string that comes from deleting up to `distance` items."""
    found = {SEPARATOR.join(sequence)}
    frontier = {tuple(sequence)}
    for _ in range(distance):
        nxt: set[tuple[str, ...]] = set()
        for item in frontier:
            for index in range(len(item)):
                shorter = item[:index] + item[index + 1 :]
                nxt.add(shorter)
                found.add(SEPARATOR.join(shorter))
        frontier = nxt
    return found


class Lexicon:
    """A Malayalam word list with a deletion index and a skeleton index."""

    def __init__(
        self,
        words: list[str],
        counts: list[int],
        max_distance: int = 2,
        index: dict[int, list[int]] | None = None,
    ) -> None:
        self.words = words
        self.counts = counts
        self.max_distance = max_distance
        self._word_ids = {word: index for index, word in enumerate(words)}
        self._skeletons = [skeleton(word) for word in words]
        self._skeleton_buckets: dict[int, dict[str, list[int]]] = {
            1: defaultdict(list),
            2: defaultdict(list),
            3: defaultdict(list),
        }
        for position, word_index in enumerate(range(len(words))):
            key = self._skeletons[word_index]
            for depth in (1, 2, 3):
                if len(key) >= depth:
                    self._skeleton_buckets[depth][key[:depth]].append(word_index)
        for depth in (1, 2, 3):
            for bucket in self._skeleton_buckets[depth].values():
                bucket.sort(key=lambda i: -self.counts[i])

        if index is None:
            self._index = self._build_index()
        else:
            self._index = index

    # --------------------------------------------------------------- building

    def _build_index(self) -> dict[int, list[int]]:
        index: dict[int, list[int]] = defaultdict(list)
        for word_index, word in enumerate(self.words):
            for variant in _deletions(graphemes(word), self.max_distance):
                index[fnv1a(variant)].append(word_index)
        return index

    # ---------------------------------------------------------------- lookups

    def __contains__(self, word: str) -> bool:
        return word in self._word_ids

    def __len__(self) -> int:
        return len(self.words)

    def count(self, word: str) -> int:
        """How often the corpus used this word."""
        index = self._word_ids.get(word)
        return self.counts[index] if index is not None else 0

    def nearest(self, word: str, limit: int = 8) -> list[str]:
        """The real words closest to a string, the best match first."""
        if not word:
            return []
        items = graphemes(word)
        found: dict[int, int] = {}
        for variant in _deletions(items, self.max_distance):
            for word_index in self._index.get(fnv1a(variant), ()):
                distance = levenshtein(items, graphemes(self.words[word_index]))
                if distance <= self.max_distance and distance < found.get(word_index, 99):
                    found[word_index] = distance

        ranked = sorted(
            found.items(),
            key=lambda pair: (pair[1], -self.counts[pair[0]], self.words[pair[0]]),
        )
        return [self.words[word_index] for word_index, _ in ranked[:limit]]

    def complete(self, prefix: str, limit: int = 8) -> list[str]:
        """The real words that start with a native prefix, the commonest first."""
        if not prefix:
            return []
        key = skeleton(prefix)
        if not key:
            return []

        depth = min(len(key), 3)
        bucket = self._skeleton_buckets[depth].get(key[:depth], [])
        out: list[str] = []
        for word_index in bucket:
            if self._skeletons[word_index].startswith(key):
                out.append(self.words[word_index])
                if len(out) >= limit:
                    break
        return out

    def expand(self, text: str, limit: int = 8) -> list[str]:
        """The real words around a string, the exact one first.

        This is what the keyboard uses to turn a model output into a list of
        words that exist.
        """
        if not text:
            return []
        out: list[str] = []
        if text in self._word_ids:
            out.append(text)
        for word in self.complete(text, limit=limit * 2):
            if word not in out:
                out.append(word)
            if len(out) >= limit:
                break
        if len(out) < limit:
            for word in self.nearest(text, limit=limit * 2):
                if word not in out:
                    out.append(word)
                if len(out) >= limit:
                    break
        return out[:limit]

    # ----------------------------------------------------------------- storage

    def save(self, directory: str | Path) -> None:
        """Write the word list and the index to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / WORD_FILE).open("w", encoding="utf-8") as handle:
            for word, count in zip(self.words, self.counts, strict=True):
                handle.write(f"{word}\t{count}\n")

        entries: list[tuple[int, int]] = []
        for key, ids in self._index.items():
            for word_index in ids:
                entries.append((key, word_index))
        entries.sort()
        with (directory / INDEX_FILE).open("wb") as handle:
            handle.write(struct.pack("<II", len(self.words), len(entries)))
            for key, word_index in entries:
                handle.write(ENTRY.pack(key, word_index))
    @classmethod
    def load(cls, directory: str | Path, max_distance: int = 2) -> "Lexicon":
        """Read a word list and its index from a directory."""
        directory = Path(directory)
        words: list[str] = []
        counts: list[int] = []
        with (directory / WORD_FILE).open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0]:
                    continue
                words.append(parts[0])
                counts.append(int(parts[1]) if len(parts) > 1 and parts[1] else 0)

        index: dict[int, list[int]] = defaultdict(list)
        raw = (directory / INDEX_FILE).read_bytes()
        count, size = struct.unpack_from("<II", raw, 0)
        for position in range(size):
            key, word_index = ENTRY.unpack_from(raw, HEADER_SIZE + position * ENTRY.size)
            index[key].append(word_index)
        if count != len(words):
            raise ValueError("the word list and the index do not match")

        return cls(words, counts, max_distance=max_distance, index=dict(index))

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, int],
        limit: int = 0,
        max_distance: int = 2,
    ) -> "Lexicon":
        """Build a lexicon from native words and their corpus counts."""
        words = sorted(counts, key=lambda word: (-counts[word], word))
        if limit:
            words = words[:limit]
        values = [counts[word] for word in words]
        return cls(words, values, max_distance=max_distance)

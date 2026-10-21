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

from model.src.metrics import letters, levenshtein

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
        self._letter_cache: dict[int, list[str]] = {}
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
            for variant in _deletions(letters(word), self.max_distance):
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
        return [text for text, _ in self.nearest_with_distance(word, limit)]

    def nearest_with_distance(self, word: str, limit: int = 8) -> list[tuple[str, int]]:
        """The real words closest to a string, with the distance to them."""
        if not word:
            return []
        items = letters(word)
        found: dict[int, int] = {}
        for variant in _deletions(items, self.max_distance):
            for word_index in self._index.get(fnv1a(variant), ()):
                distance = levenshtein(items, self._letter_list(word_index))
                if distance <= self.max_distance and distance < found.get(word_index, 99):
                    found[word_index] = distance

        ranked = sorted(
            found.items(),
            key=lambda pair: (pair[1], -self.counts[pair[0]], self.words[pair[0]]),
        )
        return [(self.words[word_index], distance) for word_index, distance in ranked[:limit]]

    def _letter_list(self, word_index: int) -> list[str]:
        """The letters of a word, kept so that the lookup stays quick."""
        cached = self._letter_cache.get(word_index)
        if cached is None:
            cached = letters(self.words[word_index])
            self._letter_cache[word_index] = cached
        return cached

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

    def expand(self, text: str, limit: int = 8) -> list[tuple[str, int]]:
        """The real words around a string, with how far they are from it.

        This is what the keyboard uses to turn a model output into a list of
        words that exist. The exact word comes first, then the words that
        start with it, then the words that are one or two edits away.
        """
        if not text:
            return []
        out: list[tuple[str, int]] = []
        seen: set[str] = set()
        if text in self._word_ids:
            out.append((text, 0))
            seen.add(text)
        for word in self.complete(text, limit * 2):
            if word in seen:
                continue
            seen.add(word)
            out.append((word, 1))
            if len(out) >= limit:
                return out
        for word, distance in self.nearest_with_distance(text, limit * 2):
            if word in seen:
                continue
            seen.add(word)
            out.append((word, distance))
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
        keep: set[str] | None = None,
        order: list[str] | None = None,
    ) -> "Lexicon":
        """Build a lexicon from native words and their counts.

        The words in `order` come first, in that order, and they are always
        kept. The limit applies to the rest, the commonest first. The words
        in `keep` are always kept as well.
        """
        keep = keep or set()
        counts = {**counts}
        for word in keep:
            counts.setdefault(word, 0)
        order = [word for word in (order or []) if word in counts]
        seen = set(order)
        rest = sorted(
            (word for word in counts if word not in seen),
            key=lambda word: (-counts[word], word),
        )
        chosen = set(order) | set(rest[:limit]) | keep
        position = {word: index for index, word in enumerate(order + rest)}
        words = sorted(chosen, key=lambda word: position[word])
        values = [counts[word] for word in words]
        return cls(words, values, max_distance=max_distance)


def count_corpus(path: str | Path) -> dict[str, int]:
    """Count the native words of a training file, ignoring the augmented rows."""
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if len(parts) > 2 and parts[2] == "adhoc":
                continue
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    return counts


def read_frequency(path: str | Path) -> list[tuple[str, int]]:
    """Read a frequency list of native words and their counts."""
    out: list[tuple[str, int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) != 2:
                continue
            out.append((parts[0], int(parts[1])))
    return out


def read_dakshina(directory: str | Path) -> dict[str, int]:
    """Read the Malayalam lexicon of the Dakshina dataset.

    The file has the native word, the romanization and the number of times
    the pair was seen.
    """
    counts: dict[str, int] = {}
    directory = Path(directory)
    for name in ("train.tsv", "valid.tsv", "test.tsv"):
        path = directory / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle):
                if number == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    count = int(parts[2])
                except ValueError:
                    continue
                counts[parts[0]] = counts.get(parts[0], 0) + count
    return counts


def build_lexicon(
    train: str | Path = "data/processed/train.tsv",
    frequency: str | Path | None = "data/raw/frequency/ml_50k.txt",
    dakshina: str | Path | None = "data/raw/dakshina",
    limit: int = 50000,
    max_distance: int = 2,
    keep: set[str] | None = None,
) -> "Lexicon":
    """Build the word list of the keyboard.

    The list comes from three sources. The corpus says which words the model
    was trained on. The frequency list of the subtitles and the Malayalam
    lexicon of the Dakshina dataset say which words people really use, and
    they decide the count of every word.

    The words of the frequency list come first, then the words of the
    Dakshina lexicon, then the commonest words of the corpus.
    """
    counts = count_corpus(train)
    order: list[str] = []
    if frequency is not None and Path(frequency).exists():
        for word, count in read_frequency(frequency):
            counts[word] = max(counts.get(word, 0), count)
            order.append(word)
    if dakshina is not None and Path(dakshina).exists():
        for word, count in sorted(read_dakshina(dakshina).items(), key=lambda item: -item[1]):
            counts[word] = max(counts.get(word, 0), count)
            if word not in order:
                order.append(word)
    return Lexicon.from_counts(
        counts,
        limit=limit,
        max_distance=max_distance,
        keep=keep,
        order=order,
    )

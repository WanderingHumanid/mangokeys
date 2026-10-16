"""Evaluation metrics for transliteration.

Character error rate is computed over grapheme clusters rather than raw
codepoints. Malayalam vowel signs and the virama are combining marks, so
counting codepoints would treat one letter as three and would make two
different models look identical.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

try:
    import regex

    _GRAPHEME = regex.compile(r"\X")
except ImportError:  # pragma: no cover
    _GRAPHEME = None

# Combining marks that belong to the previous base character.
_MALAYALAM_MARKS = "".join(chr(cp) for cp in range(0x0D3B, 0x0D4D))


def graphemes(text: str) -> list[str]:
    """Split text into grapheme clusters."""
    if _GRAPHEME is not None:
        return _GRAPHEME.findall(text)
    return _fallback_graphemes(text)


def _fallback_graphemes(text: str) -> list[str]:
    out: list[str] = []
    for char in text:
        if char in _MALAYALAM_MARKS and out:
            out[-1] += char
        else:
            out.append(char)
    return out


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


class ErrorCounter:
    """Accumulates substitution, deletion and insertion counts."""

    def __init__(self) -> None:
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.reference_length = 0
        self.exact = 0
        self.total = 0
        self.rank_hits = 0

    def update(self, reference: str, hypothesis: str) -> None:
        """Score one hypothesis against one reference."""
        ref = graphemes(reference)
        hyp = graphemes(hypothesis)
        self.total += 1
        self.reference_length += len(ref)
        if ref == hyp:
            self.exact += 1
        subs, dels, ins = _align(ref, hyp)
        self.substitutions += subs
        self.deletions += dels
        self.insertions += ins

    def update_ranking(self, reference: str, candidates: Sequence[str]) -> None:
        """Count a hit when the reference appears anywhere in the candidate list.

        This only touches the ranking counter, so it can be called next to
        update without counting the row twice.
        """
        if reference in candidates:
            self.rank_hits += 1

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        if self.reference_length == 0:
            return 0.0
        return self.errors / self.reference_length

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.exact / self.total

    @property
    def recall_at_k(self) -> float:
        if self.total == 0:
            return 0.0
        return self.rank_hits / self.total

    def summary(self) -> dict[str, float]:
        return {
            "n": self.total,
            "top1": round(self.accuracy, 4),
            "cer": round(self.cer, 4),
            "top5": round(self.recall_at_k, 4),
            "sub": self.substitutions,
            "del": self.deletions,
            "ins": self.insertions,
        }


def _align(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Count substitutions, deletions and insertions between two sequences."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        cost[i][0] = i
        back[i][0] = 1
    for j in range(1, cols):
        cost[0][j] = j
        back[0][j] = 2

    for i in range(1, rows):
        for j in range(1, cols):
            same = ref[i - 1] == hyp[j - 1]
            options = (
                (cost[i - 1][j - 1] + (0 if same else 1), 0),
                (cost[i - 1][j] + 1, 1),
                (cost[i][j - 1] + 1, 2),
            )
            best = min(options, key=lambda item: item[0])
            cost[i][j] = best[0]
            back[i][j] = best[1]

    subs = dels = ins = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 0:
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif move == 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Word level error rate over whitespace separated tokens."""
    refs = list(references)
    hyps = list(hypotheses)
    total = errors = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total += len(ref_words)
        errors += levenshtein(ref_words, hyp_words)
    return errors / total if total else 0.0

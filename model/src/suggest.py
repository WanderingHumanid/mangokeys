"""The suggestion engine of the keyboard.

The model says which word the user probably meant, the word list says which
words exist, and the language model says which of them is the natural one.
This module puts the three together and returns a ranked list.

The flow of one keystroke is:

1. the rules give an instant answer, in well under a millisecond
2. the model gives a better answer, in a few milliseconds
3. both answers are looked up in the word list, which turns them into real
   words
4. the words are ranked by how close they are to the answer, how common
   they are and how natural they sound

The rules and the model are both proposers. Nothing is thrown away before the
ranking, because the rules know things the model does not and the other way
round. A rule answer is not a word the model has never seen, and the model
resolves the ambiguities that the rules cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from model.src.lexicon import Lexicon
from model.src.ngram import CharNgram
from model.src.rules import RuleEngine


@dataclass
class Candidate:
    """One word the keyboard is considering."""

    text: str
    source: str
    model_score: float = 0.0
    lm_score: float = 0.0
    count: int = 0
    distance: int = 0
    total: float = 0.0


@dataclass
class Weights:
    """How much each signal counts in the final ranking."""

    # The score the model gave the answer.
    model: float = 1.0
    # How close the word is to the answer the rules or the model gave.
    proximity: float = 2.0
    # How natural the word sounds.
    language: float = 1.0
    # How common the word is.
    frequency: float = 0.5
    # A word that the word list does not know has to earn its place.
    unknown: float = -2.0
    # A candidate that came from the rules rather than from the model starts
    # with this score, so that the model is preferred when nothing else
    # separates the two.
    rule_penalty: float = -12.0
    limit: int = 8


@dataclass
class SuggestionEngine:
    """Ranks the words that a typed piece of Manglish could mean."""

    lexicon: Lexicon
    ngram: CharNgram | None = None
    rules: RuleEngine = field(default_factory=RuleEngine)
    weights: Weights = field(default_factory=Weights)

    def suggest(
        self,
        roman: str,
        model_candidates: list[str] | None = None,
        model_scores: list[float] | None = None,
        use_rules: bool = True,
    ) -> list[Candidate]:
        """Rank the words that a Manglish string could mean.

        The model candidates are given with the score the model gave them, so
        that the ranking can prefer the answer the model is most sure about.
        """
        roman = roman.strip()
        if not roman:
            return []

        pool: dict[str, Candidate] = {}

        for index, text in enumerate(model_candidates or []):
            if not text:
                continue
            score = 0.0
            if model_scores and index < len(model_scores):
                score = model_scores[index]
            self._offer(pool, text, "model", score)

        if use_rules:
            for text in self.rules.convert(roman)[:4]:
                self._offer(pool, text, "rules", self.weights.rule_penalty)

        out = list(pool.values())
        for candidate in out:
            self._score(candidate)
        out.sort(key=lambda item: -item.total)
        return out[: self.weights.limit]

    # ---------------------------------------------------------------- helpers

    def _offer(self, pool: dict[str, Candidate], text: str, source: str, score: float) -> None:
        """Add a word to the pool, and the real words around it."""
        existing = pool.get(text)
        if existing is not None:
            existing.model_score = max(existing.model_score, score)
            return

        pool[text] = Candidate(text=text, source=source, model_score=score)
        for word, distance in self.lexicon.expand(text, limit=4):
            if word == text:
                continue
            neighbour = pool.get(word)
            if neighbour is None:
                pool[word] = Candidate(
                    text=word, source=source, model_score=score, distance=distance
                )
            else:
                neighbour.model_score = max(neighbour.model_score, score)
                neighbour.distance = min(neighbour.distance, distance)

    def _score(self, candidate: Candidate) -> None:
        """Work out the final score of one candidate."""
        candidate.count = self.lexicon.count(candidate.text)
        if self.ngram is not None:
            candidate.lm_score = self.ngram.mean_score(candidate.text)
        candidate.total = (
            self.weights.model * candidate.model_score
            + self.weights.proximity * -candidate.distance
            + self.weights.language * candidate.lm_score
            + self.weights.frequency * math.log1p(candidate.count)
        )
        if candidate.count == 0:
            candidate.total += self.weights.unknown


def complete_prefix(
    lexicon: Lexicon,
    ngram: CharNgram | None,
    prefix: str,
    limit: int = 8,
) -> list[str]:
    """The words that start with a native prefix, the best ones first."""
    words = lexicon.complete(prefix, limit=limit * 4)
    if not words or ngram is None:
        return words[:limit]
    ranked = sorted(words, key=lambda word: (-lexicon.count(word), ngram.mean_score(word)))
    return ranked[:limit]

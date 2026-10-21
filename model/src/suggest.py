"""The suggestion engine of the keyboard.

The model says which word the user probably meant, the word list says which
words exist, and the language model says which of them is the natural one.
This module puts the three together and returns a ranked list.

What the measurements say about the three signals.

The model is the best judge of what the user meant. It gets the word right
more often than anything the word list can offer, so its answer comes first
and the word list never replaces it.

The word list is a good judge of how a word is written. On the words it
knows, snapping the answer to the nearest word in the list is worth fifteen
points. On the words it does not know, snapping makes things worse, because
the nearest word is then a different word. So the words around the answer are
offered as alternatives, the way a keyboard offers variations.

A word that extends the answer is a special case. When the user has typed
half a word, the model says "ആശംസക" and the word list says "ആശംസകൾ" is a
word that starts with it. That is the completion, and it is the reason the
model is trained on prefixes as well as on whole words.
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
    known: bool = False
    completion: bool = False
    variation: bool = False
    total: float = 0.0


@dataclass
class Weights:
    """How much each signal counts in the final ranking.

    These are the values that `scripts/tune_suggestions.py` worked out on the
    validation split.
    """

    # The score the model gave the answer, relative to its best answer.
    model: float = 1.0
    # A word that the word list knows.
    known: float = 4.0
    # A word that extends what the model said.
    completion: float = 0.0
    # A word that only resembles what the model said.
    variation: float = -2.0
    # How natural the word sounds.
    language: float = 0.0
    # How common the word is.
    frequency: float = 0.0
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
        completion_candidates: list[str] | None = None,
        use_rules: bool = True,
    ) -> list[Candidate]:
        """Rank the words that a Manglish string could mean.

        The model candidates are given with the score the model gave them, so
        that the ranking can prefer the answer the model is most sure about.
        The completion candidates are the words that start with the answer of
        the model that was trained on prefixes.
        """
        roman = roman.strip()
        if not roman:
            return []

        pool: dict[str, Candidate] = {}
        best = model_scores[0] if model_scores else 0.0

        for index, text in enumerate(model_candidates or []):
            if not text:
                continue
            score = 0.0
            if model_scores and index < len(model_scores):
                score = model_scores[index] - best
            self._add(pool, text, "model", score)

        for text in completion_candidates or []:
            if not text:
                continue
            candidate = pool.get(text)
            if candidate is None:
                candidate = Candidate(text=text, source="completion", model_score=0.0)
                pool[text] = candidate
            candidate.completion = True

        if use_rules:
            for text in self.rules.convert(roman)[:4]:
                self._add(pool, text, "rules", self.weights.rule_penalty)

        out = list(pool.values())
        for candidate in out:
            self._score(candidate)
        out.sort(key=lambda item: -item.total)
        return out[: self.weights.limit]

    # ---------------------------------------------------------------- helpers

    def _add(self, pool: dict[str, Candidate], text: str, source: str, score: float) -> None:
        """Add an answer and the real words around it."""
        candidate = pool.get(text)
        if candidate is None:
            candidate = Candidate(text=text, source=source, model_score=score)
            pool[text] = candidate
        else:
            candidate.model_score = max(candidate.model_score, score)

        # The answer is a real word when the word list knows it. The words
        # that the model itself proposed get the bonus, because the model is
        # the better judge of what the user meant.
        if text in self.lexicon and source == "model":
            candidate.known = True

        # The words that start with the answer are the completion. They are a
        # different thing from the words that merely resemble it, and they
        # are the answer only when the model did not give a whole word.
        if not candidate.known:
            for word in self.lexicon.complete(text, limit=3):
                if word == text:
                    continue
                neighbour = pool.get(word)
                if neighbour is None:
                    neighbour = Candidate(
                        text=word,
                        source=source,
                        model_score=score,
                        completion=True,
                    )
                    pool[word] = neighbour
                else:
                    neighbour.model_score = max(neighbour.model_score, score)
                    neighbour.completion = True

        # The words that resemble the answer are the variations.
        for word, _ in self.lexicon.nearest_with_distance(text, limit=3):
            if word == text:
                continue
            neighbour = pool.get(word)
            if neighbour is None:
                neighbour = Candidate(
                    text=word,
                    source=source,
                    model_score=score,
                    variation=True,
                )
                pool[word] = neighbour
            else:
                neighbour.model_score = max(neighbour.model_score, score)
                neighbour.variation = True

    def _score(self, candidate: Candidate) -> None:
        """Work out the final score of one candidate."""
        candidate.count = self.lexicon.count(candidate.text)
        if self.ngram is not None:
            candidate.lm_score = self.ngram.mean_score(candidate.text)
        candidate.total = (
            self.weights.model * candidate.model_score
            + self.weights.known * (1.0 if candidate.known else 0.0)
            + self.weights.completion * (1.0 if candidate.completion else 0.0)
            + self.weights.variation * (1.0 if candidate.variation else 0.0)
            + self.weights.language * candidate.lm_score
            + self.weights.frequency * math.log1p(candidate.count)
        )

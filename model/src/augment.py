"""Augmentation that manufactures the typing style real Manglish users employ.

Public transliteration corpora are written in careful, standard spelling.
Real users drop inherent vowels, collapse gemination, use non standard digraphs
and ignore the retroflex distinction entirely. The IndoNLP 2025 shared task
measured the cost of that shift: character error rate goes from 7.4 percent on
careful typing to 22.7 percent on real typing.

Each rule below rewrites the roman side of a pair while leaving the Malayalam
target untouched. That is deliberate. A user who types "ta" for a word whose
correct spelling is "Ta" has not made a random error, they have created a
genuine ambiguity. Training on both spellings teaches the model to use context
to pick the right letter, which is exactly the skill that is missing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

VOWEL_LETTERS = set("aeiou")
CONSONANT_LETTERS = set("bcdfghjklmnpqrstvwxyz")
# Letters that may follow a vowel to form a diphthong.
DIPHTHONG_TAILS = set("aiu")

# Doubles that are unambiguous when collapsed. "tt" and "nn" are excluded
# because they are also how റ്റ് and ണ്ണ് are written.
SAFE_DOUBLES = set("kpcsmgbdvyhr")

# Retroflex letters in the common romanization scheme use uppercase.
RETROFLEX = {"t": "T", "th": "Th", "d": "D", "dh": "Dh", "n": "N"}

# Alternate spellings that mean the same thing.
SYNONYMS = {
    "za": "Sa",
    "Sa": "za",
    "shh": "sh",
    "w": "v",
    "f": "ph",
    "c": "k",
    "q": "kh",
    "rr": "R",
    "R": "rr",
}


@dataclass
class AugmentConfig:
    """Per rule probabilities. All of them default to conservative values."""

    drop_inherent_vowels: float = 0.35
    drop_short_i: float = 0.20
    drop_final_u: float = 0.15
    collapse_gemination: float = 0.15
    geminate_final_consonant: float = 0.10
    drop_aspiration: float = 0.15
    add_aspiration: float = 0.10
    swap_retroflex_case: float = 0.20
    simplify_palatal: float = 0.10
    reduce_nasal_cluster: float = 0.10
    use_synonym: float = 0.08
    drop_final_consonant: float = 0.05
    join_compound: float = 0.05
    rule_weights: dict[str, float] = field(default_factory=dict)


class ManglishAugmenter:
    """Applies style rules to the roman side of a transliteration pair."""

    def __init__(self, config: AugmentConfig | None = None, seed: int = 0) -> None:
        self.config = config or AugmentConfig()
        self.rng = random.Random(seed)
        self._rules = [
            self._drop_inherent_vowels,
            self._drop_short_i,
            self._drop_final_u,
            self._collapse_gemination,
            self._geminate_final_consonant,
            self._drop_aspiration,
            self._add_aspiration,
            self._swap_retroflex_case,
            self._simplify_palatal,
            self._reduce_nasal_cluster,
            self._use_synonym,
            self._drop_final_consonant,
            self._join_compound,
        ]

    def augment(self, roman: str) -> tuple[str, list[str]]:
        """Return a rewritten roman string and the list of rules applied."""
        text = roman
        applied: list[str] = []
        for rule in self._rules:
            if not self._should_apply(rule):
                continue
            new_text = rule(text)
            if new_text != text:
                applied.append(rule.__name__.lstrip("_"))
                text = new_text
        return text, applied

    def _should_apply(self, rule) -> bool:
        name = rule.__name__.lstrip("_")
        weight = self.config.rule_weights.get(name, 1.0)
        probability = getattr(self.config, name, 0.0) * weight
        return self.rng.random() < probability

    # ------------------------------------------------------------------ rules

    def _drop_inherent_vowels(self, text: str) -> str:
        """Drop the inherent a that sits between two consonants.

        In Malayalam a consonant that carries a virama has no vowel of its
        own, so the roman spelling may or may not write the a. "sundari" and
        "sundri" both mean സുന്ദരി, and the model has to work out which
        consonant the a belonged to from the rest of the word.
        """
        chars = list(text)
        out: list[str] = []
        for index, char in enumerate(chars):
            if char == "a" and self._is_droppable_a(chars, index):
                continue
            out.append(char)
        return "".join(out)

    def _is_droppable_a(self, chars: list[str], index: int) -> bool:
        if index == 0 or index + 1 >= len(chars):
            return False
        previous, following = chars[index - 1], chars[index + 1]
        if previous not in CONSONANT_LETTERS or following not in CONSONANT_LETTERS:
            return False
        if following in DIPHTHONG_TAILS:
            return False
        if following == "m" and index + 2 >= len(chars):
            return False
        return True

    def _drop_short_i(self, text: str) -> str:
        """Drop the i vowel sign. "ni" becomes "n", as in നിങ്ങൾക്കും."""
        for index in range(len(text) - 1, 0, -1):
            if text[index] != "i":
                continue
            if text[index - 1] not in CONSONANT_LETTERS:
                continue
            if index > 0 and text[index - 1] == "a":
                continue
            return text[:index] + text[index + 1 :]
        return text

    def _drop_final_u(self, text: str) -> str:
        """Drop a word final u sign. "cheythu" becomes "cheyth"."""
        if len(text) < 4 or not text.endswith("u"):
            return text
        if text[-2] not in CONSONANT_LETTERS:
            return text
        return text[:-1]

    def _collapse_gemination(self, text: str) -> str:
        """Halve a doubled consonant. "kattu" becomes "katu"."""
        for letter in sorted(SAFE_DOUBLES):
            if letter * 2 in text:
                return text.replace(letter * 2, letter, 1)
        return text

    def _geminate_final_consonant(self, text: str) -> str:
        """Write a word final chillu with a double letter. "aval" becomes "avall"."""
        if len(text) < 3 or text[-1] not in "lnr":
            return text
        return text + text[-1]

    def _drop_aspiration(self, text: str) -> str:
        """Remove the h that marks aspiration. "kha" becomes "ka"."""
        for digraph in ("chh", "ch", "kh", "gh", "thh", "dh", "ph", "bh", "th"):
            if digraph in text:
                return text.replace(digraph, digraph[0], 1)
        return text

    def _add_aspiration(self, text: str) -> str:
        """Add aspiration. "ka" becomes "kha", "sa" becomes "sha"."""
        for digraph, aspirated in (("sa", "sha"), ("ca", "cha"), ("ka", "kha"), ("ta", "tha")):
            if digraph in text:
                return text.replace(digraph, aspirated, 1)
        return text

    def _swap_retroflex_case(self, text: str) -> str:
        """Change the case of a retroflex consonant.

        The letter stays the same, only the case changes, so the meaning of the
        pair is untouched. The model learns that the case carries little
        information and has to look at the whole word.
        """
        for lower, upper in RETROFLEX.items():
            if lower in text:
                return text.replace(lower, upper, 1)
        return text

    def _simplify_palatal(self, text: str) -> str:
        """Drop the h from a palatal aspirate and shorten the cluster."""
        for digraph, short in (("chh", "ch"), ("ch", "c"), ("nj", "n"), ("jh", "j")):
            if digraph in text:
                return text.replace(digraph, short, 1)
        return text

    def _reduce_nasal_cluster(self, text: str) -> str:
        """Shorten a nasal cluster. "nth" becomes "nt"."""
        for cluster, short in (("nth", "nt"), ("ndh", "nd"), ("nch", "nc")):
            if cluster in text:
                return text.replace(cluster, short, 1)
        return text

    def _use_synonym(self, text: str) -> str:
        """Swap in an alternate spelling of the same letter."""
        for source, target in SYNONYMS.items():
            if source in text:
                return text.replace(source, target, 1)
        return text

    def _drop_final_consonant(self, text: str) -> str:
        """Drop the last consonant of a long word."""
        if len(text) < 5 or text[-1] not in CONSONANT_LETTERS:
            return text
        return text[:-1]

    def _join_compound(self, text: str) -> str:
        """Remove a space so that a compound is typed as one run."""
        if " " not in text:
            return text
        parts = text.split(" ")
        if len(parts) < 2 or not all(parts):
            return text
        index = self.rng.randrange(len(parts) - 1)
        parts[index] = parts[index] + parts[index + 1]
        del parts[index + 1]
        return " ".join(parts)


def augment_pairs(
    pairs: list[dict],
    augmenter: ManglishAugmenter,
    copies: int = 1,
) -> list[dict]:
    """Produce augmented copies of a list of pairs.

    The original pairs are returned unchanged. Augmented copies carry a style
    tag so that the effect of augmentation can be measured rather than assumed.
    """
    out: list[dict] = []
    for pair in pairs:
        out.append({**pair, "style": "standard"})
        for _ in range(copies):
            roman, applied = augmenter.augment(pair["roman"])
            if not applied or roman == pair["roman"]:
                continue
            out.append(
                {
                    "roman": roman,
                    "native": pair["native"],
                    "style": "adhoc",
                    "rules": ",".join(applied),
                }
            )
    return out


def split_sentences(text: str) -> list[str]:
    """Split a romanized sentence into whitespace separated words."""
    return [word for word in re.split(r"\s+", text.strip()) if word]

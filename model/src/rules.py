"""A rule based Manglish to Malayalam converter.

This is the baseline the neural model has to beat. It follows the romanization
that the training corpus actually uses, which is close to the common Malayalam
keyboard scheme: a table of consonants, a table of vowels, a small set of
clusters, and a handful of context rules for the places where Malayalam
orthography is not a simple concatenation.

The engine cannot resolve ambiguity. A consonant in a cluster and the same
consonant standing alone look the same in the roman text, so the engine cannot
tell സുന്ദരി from സുന്ദ്രി. It produces several candidates instead and lets a
ranker choose, which is how the shipped keyboards work.

The primary reading of each letter is the one the corpus uses most often. The
rest are offered as alternatives. The roman side barely uses the uppercase
retroflex convention: T appears in six percent of rows, D in two and N in
two, while R and S never appear in capitals at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VIRAMA = "്"
ANUSVARA = "ം"
ZWNJ = "‌"

INDEPENDENT_VOWELS = {
    "a": "അ",
    "aa": "ആ",
    "i": "ഇ",
    "ii": "ഈ",
    "u": "ഉ",
    "uu": "ഊ",
    "e": "എ",
    "E": "ഏ",
    "ai": "ഐ",
    "o": "ഒ",
    "O": "ഓ",
    "au": "ഔ",
}

VOWEL_SIGNS = {
    "a": "",
    "aa": "ാ",
    "i": "ി",
    "ii": "ീ",
    "u": "ു",
    "uu": "ൂ",
    "e": "െ",
    "ee": "േ",
    "E": "േ",
    "ai": "ൈ",
    "o": "ൊ",
    "oo": "ോ",
    "O": "ോ",
    "au": "ൗ",
}

CONSONANTS = {
    "k": "ക",
    "kh": "ഖ",
    "g": "ഗ",
    "gh": "ഘ",
    "ng": "ങ",
    "ch": "ച",
    "chh": "ഛ",
    "j": "ജ",
    "jh": "ഝ",
    "nj": "ഞ",
    "T": "ട",
    "Th": "ഠ",
    "D": "ഡ",
    "Dh": "ഢ",
    "N": "ണ",
    "t": "ത",
    "th": "ത",
    "thh": "ഥ",
    "d": "ദ",
    "dh": "ധ",
    "n": "ന",
    "p": "പ",
    "ph": "ഫ",
    "b": "ബ",
    "bh": "ഭ",
    "m": "മ",
    "y": "യ",
    "r": "ര",
    "l": "ല",
    "v": "വ",
    "w": "വ",
    "sh": "ശ",
    "Sh": "ഷ",
    "s": "സ",
    "h": "ഹ",
    "L": "ള",
    "zh": "ഴ",
    "R": "റ",
    "rr": "റ",
}

CLUSTERS = {
    "nt": "ന്റ",
    "nk": "ങ്ക",
    "nch": "ഞ്ച",
    "ksh": "ക്ഷ",
    "nd": "ന്ദ",
    "mb": "മ്പ",
    "nth": "ന്ത",
}

# Letters that can end a word as a chillu.
CHILLUS = {
    "ണ": "ൺ",
    "ന": "ൻ",
    "ര": "ർ",
    "ല": "ൽ",
    "ള": "ൾ",
    "ക": "ൿ",
}

# Alternate spellings that produce the same sound. The primary mapping of each
# roman letter is the one the corpus uses most often, and the alternatives are
# the readings a person could have meant instead. Measured from the training
# corpus: t is ത 65 percent of the time and ട 35 percent, s is സ 60 percent
# with ശ and ഷ making up the rest, and so on.
ALTERNATIVES = {
    "t": ["ട", "റ", "റ്റ"],
    "th": ["ഥ", "ഠ"],
    "d": ["ഡ", "ധ"],
    "dh": ["ഢ", "ഥ"],
    "n": ["ണ"],
    "l": ["ള"],
    "r": ["റ"],
    "s": ["ശ", "ഷ"],
    "sh": ["ഷ", "സ"],
    "ng": ["ങ്ങ"],
    "kk": ["ക്ക"],
    "nd": ["ണ്ട"],
    "nth": ["ന്ത", "ന്റ"],
    "T": ["ത"],
    "Th": ["ഥ", "ത"],
    "D": ["ദ"],
    "Dh": ["ധ", "ദ"],
    "N": ["ന"],
    "L": ["ല"],
    "R": ["ര"],
    "Sh": ["ശ", "സ"],
}


@dataclass
class Segment:
    """One syllable piece of a parsed word."""

    kind: str
    text: str
    key: str = ""
    vowel_key: str | None = None
    vowel_sign: str | None = None
    doubled: bool = False
    base: str = ""


@dataclass
class RuleEngine:
    """Deterministic Manglish to Malayalam converter."""

    max_candidates: int = 12
    _cluster_keys: list[str] = field(default_factory=list)
    _consonant_keys: list[str] = field(default_factory=list)
    _vowel_keys: list[str] = field(default_factory=list)
    _sign_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._cluster_keys = sorted(CLUSTERS, key=len, reverse=True)
        self._consonant_keys = sorted(CONSONANTS, key=len, reverse=True)
        self._vowel_keys = sorted(INDEPENDENT_VOWELS, key=len, reverse=True)
        self._sign_keys = sorted(VOWEL_SIGNS, key=len, reverse=True)

    def convert(self, text: str) -> list[str]:
        """Return candidate Malayalam spellings, best guess first."""
        text = text.strip()
        if not text:
            return []
        segments = self._parse(text)
        primary = self._render(segments)
        results = [primary]

        for index, segment in enumerate(segments):
            if segment.kind != "consonant":
                continue
            for alternative in ALTERNATIVES.get(segment.key, []):
                if alternative == segment.base:
                    continue
                candidate = self._render(self._replace(segments, index, segment, base=alternative))
                if candidate not in results:
                    results.append(candidate)

        return results[: self.max_candidates]

    def convert_top(self, text: str) -> str:
        candidates = self.convert(text)
        return candidates[0] if candidates else ""

    # ---------------------------------------------------------------- parsing

    def _parse(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        index = 0
        while index < len(text):
            cluster = self._match(text, index, self._cluster_keys)
            if cluster:
                sign, sign_key = self._match_sign(text, index + len(cluster))
                segments.append(
                    Segment(
                        kind="consonant",
                        text=CLUSTERS[cluster],
                        key=cluster,
                        base=CLUSTERS[cluster],
                        vowel_key=sign_key,
                        vowel_sign=sign,
                    )
                )
                index += len(cluster) + (len(sign_key) if sign_key else 0)
                continue

            consonant = self._match(text, index, self._consonant_keys)
            if consonant:
                base = CONSONANTS[consonant]
                doubled = len(consonant) == 1 and text[index + 1 : index + 2] == consonant
                advance = 2 if doubled else len(consonant)
                if doubled:
                    base = base + VIRAMA + base
                sign, sign_key = self._match_sign(text, index + advance)
                segments.append(
                    Segment(
                        kind="consonant",
                        text=base,
                        key=consonant,
                        base=base,
                        doubled=doubled,
                        vowel_key=sign_key,
                        vowel_sign=sign,
                    )
                )
                index += advance + (len(sign_key) if sign_key else 0)
                continue

            vowel = self._match(text, index, self._vowel_keys)
            if vowel:
                segments.append(
                    Segment(
                        kind="vowel",
                        text=INDEPENDENT_VOWELS[vowel],
                        key=vowel,
                        vowel_key=vowel,
                    )
                )
                index += len(vowel)
                continue

            segments.append(Segment(kind="other", text=text[index]))
            index += 1

        return segments

    @staticmethod
    def _match(text: str, index: int, keys: list[str]) -> str | None:
        for key in keys:
            if text.startswith(key, index):
                return key
        return None

    def _match_sign(self, text: str, index: int) -> tuple[str, str | None]:
        key = self._match(text, index, self._sign_keys)
        if key is None:
            return "", None
        return VOWEL_SIGNS[key], key

    # -------------------------------------------------------------- rendering

    def _render(self, segments: list[Segment]) -> str:
        out: list[str] = []
        last = len(segments) - 1
        for index, segment in enumerate(segments):
            if segment.kind == "vowel":
                out.append(segment.text)
                continue
            if segment.kind == "other":
                out.append(segment.text)
                continue

            base = segment.base
            if segment.vowel_key is not None:
                out.append(base + (segment.vowel_sign or ""))
                continue

            # A consonant with no vowel after it.
            if index < last:
                out.append(base)
                if segments[index + 1].kind == "consonant":
                    out.append(VIRAMA)
                continue

            if not segment.doubled and base == "മ":
                out.append(ANUSVARA)
                continue
            if not segment.doubled and base in CHILLUS:
                out.append(CHILLUS[base])
                continue
            out.append(base + VIRAMA)

        return "".join(out)

    @staticmethod
    def _replace(
        segments: list[Segment],
        index: int,
        segment: Segment,
        base: str | None = None,
    ) -> list[Segment]:
        out = list(segments)
        copy = Segment(**{**segment.__dict__})
        if base is not None:
            copy.base = base
            copy.text = base
        out[index] = copy
        return out


def load_rules(path: str | Path) -> RuleEngine:
    """Load a rule engine from a json file of overrides."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, table in (("consonants", CONSONANTS), ("vowel_signs", VOWEL_SIGNS)):
        if name in data:
            table.update(data[name])
    return RuleEngine()

"""Unicode normalization for Malayalam text.

Malayalam has a dual encoding for the vowel-less chillu forms. The legacy
Unicode 5.0 spelling is consonant + virama + anusvara, while the modern
spelling is a single atomic codepoint. Fonts render both, but an input method
should only ever emit the atomic forms, so we rewrite them here.

The ZWJ and ZWNJ forms carry meaning and must not be touched. A chillu is
consonant + virama + ZWJ, an explicit virama is consonant + virama + ZWNJ, and
those are different words.
"""

from __future__ import annotations

import re
import unicodedata

VIRAMA = "്"
ANUSVARA = "ം"
ZWJ = "‍"
ZWNJ = "‌"

# Legacy chillu sequences mapped to their atomic modern equivalent.
LEGACY_CHILLU = {
    "ണ": "ൺ",
    "ന": "ൻ",
    "ര": "ർ",
    "ല": "ൽ",
    "ള": "ൾ",
    "ക": "ൿ",
}

# The ZWJ spelling of the same forms.
ZWJ_CHILLU = {cons: chillu for cons, chillu in LEGACY_CHILLU.items()}

# ന്റ is encoded as chillu-n + virama + റ in modern Unicode but as ന + ് + റ in
# the legacy spelling.
NRA_LEGACY = "ന" + VIRAMA + "റ"
NRA_MODERN = "ൻ" + VIRAMA + "റ"

_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
    "«": '"',
    "»": '"',
}

_DASHES = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍⁠﻿"), None)


def normalize_malayalam(text: str) -> str:
    """Normalize native Malayalam text to the modern canonical form."""
    text = unicodedata.normalize("NFC", text)
    text = _rewrite_legacy_chillu(text)
    text = text.replace(NRA_LEGACY, NRA_MODERN)
    return text


def normalize_roman(text: str) -> str:
    """Normalize the Manglish source side.

    Case is preserved because it carries meaning: in the common romanization
    scheme uppercase T, D and N stand for the retroflex consonants.
    """
    text = unicodedata.normalize("NFC", text)
    for bad, good in _QUOTES.items():
        text = text.replace(bad, good)
    for bad, good in _DASHES.items():
        text = text.replace(bad, good)
    text = text.translate(_ZERO_WIDTH)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _rewrite_legacy_chillu(text: str) -> str:
    """Replace legacy and ZWJ chillu spellings with atomic codepoints."""
    for cons, chillu in LEGACY_CHILLU.items():
        text = text.replace(cons + VIRAMA + ANUSVARA, chillu)
    for cons, chillu in ZWJ_CHILLU.items():
        text = text.replace(cons + VIRAMA + ZWJ, chillu)
    return text


def has_legacy_chillu(text: str) -> bool:
    """True when the text still uses a non atomic chillu spelling."""
    return normalize_malayalam(text) != unicodedata.normalize("NFC", text)

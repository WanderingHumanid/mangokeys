"""Tests for the Malayalam unicode normalization."""

from model.src.normalize import (
    has_legacy_chillu,
    normalize_malayalam,
    normalize_roman,
)

# Legacy chillu spellings and the atomic forms they should become.
LEGACY_CASES = [
    ("ന" + "്" + "ം", "ൻ"),
    ("ണ" + "്" + "ം", "ൺ"),
    ("ര" + "്" + "ം", "ർ"),
    ("ല" + "്" + "ം", "ൽ"),
    ("ള" + "്" + "ം", "ൾ"),
    ("ക" + "്" + "ം", "ൿ"),
    ("ന" + "്" + "‍", "ൻ"),
    ("ല" + "്" + "‍", "ൽ"),
]


def test_legacy_chillu_becomes_atomic():
    for legacy, atomic in LEGACY_CASES:
        assert normalize_malayalam(legacy) == atomic


def test_explicit_virama_is_preserved():
    # ന്ം with ZWNJ means a dead consonant, not a chillu.
    text = "ന" + "്" + "‌"
    assert normalize_malayalam(text) == text


def test_nra_is_rewritten():
    assert normalize_malayalam("ന" + "്" + "റ") == "ൻ" + "്" + "റ"


def test_chillu_inside_a_word():
    text = "അ" + "വ" + "ന" + "്" + "ം"
    assert normalize_malayalam(text) == "അവൻ"


def test_already_atomic_is_untouched():
    text = "നിങ്ങൾക്കും സുഖമാണോ"
    assert normalize_malayalam(text) == text
    assert not has_legacy_chillu(text)


def test_roman_case_is_preserved():
    # Uppercase T, D and N are the retroflex consonants.
    assert normalize_roman("Ta") == "Ta"
    assert normalize_roman("th") == "th"


def test_roman_quotes_and_dashes():
    assert normalize_roman("“hello”") == '"hello"'
    assert normalize_roman("a—b") == "a-b"


def test_roman_whitespace_collapses():
    assert normalize_roman("  ente   peru  ") == "ente peru"

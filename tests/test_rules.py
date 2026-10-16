"""Tests for the rule based converter."""

from model.src.rules import RuleEngine


def engine() -> RuleEngine:
    return RuleEngine()


def test_simple_words():
    eng = engine()
    assert eng.convert_top("ka") == "ക"
    assert eng.convert_top("kaa") == "കാ"
    assert eng.convert_top("kam") == "കം"
    assert eng.convert_top("av") == "അവ്"


def test_independent_vowel():
    eng = engine()
    assert eng.convert_top("a") == "അ"
    assert eng.convert_top("aa") == "ആ"


def test_consonant_with_sign():
    eng = engine()
    assert eng.convert_top("ki") == "കി"
    assert eng.convert_top("ke") == "കെ"


def test_gemination():
    eng = engine()
    assert eng.convert_top("kka") == "ക്ക"


def test_cluster():
    eng = engine()
    assert eng.convert_top("nta") == "ന്റ"


def test_final_consonant_becomes_chillu():
    eng = engine()
    assert eng.convert_top("avan") == "അവൻ"
    assert eng.convert_top("aval") == "അവൽ"


def test_alternatives_are_offered():
    eng = engine()
    candidates = eng.convert("sundari")
    assert len(candidates) > 1
    assert candidates[0] in candidates


def test_rule_engine_cannot_resolve_ambiguity():
    # This is the whole reason a learned ranker is worth building.
    eng = engine()
    candidates = eng.convert("sundari")
    assert len(candidates) > 1


def test_empty_input():
    eng = engine()
    assert eng.convert("") == []
    assert eng.convert_top("   ") == ""

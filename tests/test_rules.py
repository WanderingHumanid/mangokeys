"""Tests for the rule based converter."""

from model.src.rules import CONSONANTS, RuleEngine


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


def test_retroflex_and_dental_are_distinct():
    eng = engine()
    assert eng.convert_top("Ta") == "ട"
    assert eng.convert_top("ta") == "ത"
    assert eng.convert_top("Tha") == "ഠ"
    assert eng.convert_top("tha") == "ത"
    assert eng.convert_top("thha") == "ഥ"
    assert eng.convert_top("Da") == "ഡ"
    assert eng.convert_top("da") == "ദ"
    assert eng.convert_top("Dha") == "ഢ"
    assert eng.convert_top("dha") == "ധ"
    assert eng.convert_top("Na") == "ണ"
    assert eng.convert_top("na") == "ന"


def test_consonant_table_has_no_duplicate_keys():
    # A duplicate key in the table silently drops one of the two mappings.
    assert len(CONSONANTS) == len(set(CONSONANTS))
    assert CONSONANTS["dh"] == "ധ"
    assert CONSONANTS["thh"] == "ഥ"


def test_retroflex_alternatives_are_offered():
    eng = engine()
    candidates = eng.convert("Ta")
    assert "ത" in candidates


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
    assert len(eng.convert("sundari")) > 1


def test_empty_input():
    eng = engine()
    assert eng.convert("") == []
    assert eng.convert_top("   ") == ""

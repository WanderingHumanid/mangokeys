"""Tests for the suggestion engine."""

from model.src.lexicon import Lexicon
from model.src.ngram import CharNgram
from model.src.suggest import SuggestionEngine, Weights


def lexicon() -> Lexicon:
    words = ["ഒരു", "എന്റെ", "എൻ്റെ", "കേരളം", "പേരു", "ആന", "ക്സസ്സ്"]
    counts = [100, 50, 2, 30, 10, 20, 1]
    return Lexicon(words, counts, max_distance=2)


def engine() -> SuggestionEngine:
    return SuggestionEngine(lexicon=lexicon(), ngram=CharNgram.train(["ഒരു", "ഒരു", "എന്റെ"]))


def test_the_model_answer_comes_first():
    ranked = engine().suggest("ente", ["എൻ്റെ"], [-1.0])
    assert ranked[0].text == "എൻ്റെ"


def test_a_real_word_beats_a_word_the_list_does_not_know():
    ranked = engine().suggest("ente", ["ക്സ്സ്", "ഒരു"], [-1.0, -2.0])
    assert ranked[0].text == "ഒരു"


def test_the_variations_come_after_the_answer():
    ranked = engine().suggest("ente", ["ക്സസ്സ്"], [-1.0])
    assert ranked[0].text == "ക്സസ്സ്"
    assert "എന്റെ" in [item.text for item in ranked]


def test_an_empty_word_gives_nothing():
    assert engine().suggest("") == []


def test_the_limit_is_respected():
    weights = Weights(limit=2)
    engine = SuggestionEngine(lexicon=lexicon(), weights=weights)
    assert len(engine.suggest("ente", ["ക്സസ്സ്"], [-1.0])) <= 2


def test_the_completion_is_offered_when_the_answer_is_not_a_word():
    ranked = engine().suggest("oru", ["ഒരു"], [-1.0])
    assert "ഒരു" in [item.text for item in ranked]

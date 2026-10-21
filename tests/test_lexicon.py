"""Tests for the Malayalam word list."""

from model.src.lexicon import Lexicon, skeleton


def lexicon() -> Lexicon:
    words = ["കേരളം", "കേരളി", "പേരു", "എൻ്റെ", "സുന്ദരി", "കാലം", "ആന"]
    counts = [30, 12, 25, 20, 18, 40, 50]
    return Lexicon(words, counts, max_distance=2)


def test_skeleton_drops_the_marks():
    assert skeleton("കേരളം") == "കരള"
    assert skeleton("കെരളം") == "കരള"
    assert skeleton("ആന") == "ന"


def test_skeleton_keeps_the_letters_apart():
    assert skeleton("ശ") != skeleton("ഷ")
    assert skeleton("ട") != skeleton("ത")


def test_nearest_finds_the_word_with_a_wrong_vowel():
    assert lexicon().nearest("കെരളം", limit=1) == ["കേരളം"]


def test_nearest_finds_the_word_with_a_missing_letter():
    assert lexicon().nearest("കേര", limit=1) == ["കേരളം"]


def test_nearest_keeps_an_exact_word_first():
    assert lexicon().nearest("പേരു", limit=1) == ["പേരു"]


def test_complete_finds_the_words_under_a_prefix():
    found = lexicon().complete("കേരള")
    assert "കേരളം" in found
    assert "കേരളി" in found


def test_complete_ignores_a_wrong_vowel_sign():
    assert "കേരളം" in lexicon().complete("കെരള")


def test_complete_returns_the_commonest_first():
    assert lexicon().complete("കേരള")[0] == "കേരളം"


def test_expand_returns_the_exact_word_first():
    assert lexicon().expand("കേരളം")[0] == ("കേരളം", 0)


def test_expand_fixes_a_wrong_vowel():
    assert "കേരളം" in [word for word, _ in lexicon().expand("കെരളം")]


def test_count_reports_the_corpus_frequency():
    assert lexicon().count("കേരളം") == 30
    assert lexicon().count("ഇല്ല") == 0


def test_contains():
    assert "പേരു" in lexicon()
    assert "ഇല്ല" not in lexicon()


def test_saves_and_loads(tmp_path):
    original = lexicon()
    original.save(tmp_path)
    loaded = Lexicon.load(tmp_path)
    assert loaded.words == original.words
    assert loaded.counts == original.counts
    assert loaded.nearest("കെരളം", limit=1) == ["കേരളം"]


def test_from_counts_keeps_the_commonest_words():
    built = Lexicon.from_counts({"ആ": 5, "ഇ": 9, "ഈ": 1}, limit=2)
    assert built.words[0] == "ഇ"
    assert len(built) == 2

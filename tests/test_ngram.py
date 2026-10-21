"""Tests for the character language model."""

from model.src.ngram import CharNgram


def model() -> CharNgram:
    words = ["കേരളം"] * 5 + ["കേരളി"] + ["കേരളത്തിൽ"] + ["പേരു"] * 3 + ["ആന"] * 4
    return CharNgram.train(words, order=3, alpha=0.1)


def test_a_word_it_has_seen_scores_better_than_an_unknown_one():
    trained = model()
    assert trained.score("കേരളം") > trained.score("ക്സസ്സ്")


def test_a_common_word_scores_better_than_a_rare_one():
    trained = model()
    assert trained.score("പേരു") > trained.score("കേരളി")


def test_the_mean_score_does_not_favour_long_words():
    trained = model()
    assert trained.mean_score("കേരളം") > trained.mean_score("ക്സസ്സ്")


def test_saves_and_loads(tmp_path):
    trained = model()
    trained.save(tmp_path / "ngram.txt")
    loaded = CharNgram.load(tmp_path / "ngram.txt")
    assert loaded.order == trained.order
    assert abs(loaded.score("കേരളം") - trained.score("കേരളം")) < 1e-9
    assert abs(loaded.score("കേരളി") - trained.score("കേരളി")) < 1e-9


def test_an_empty_word_scores_zero():
    assert model().mean_score("") == 0.0

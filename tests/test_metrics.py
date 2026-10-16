"""Tests for the evaluation metrics."""

from model.src.metrics import ErrorCounter, graphemes, levenshtein, word_error_rate


def test_graphemes_keep_matra_attached():
    # കേരളം is five codepoints but three grapheme clusters.
    assert len(graphemes("കേരളം")) == 3
    assert graphemes("കേ") == ["കേ"]


def test_levenshtein():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("abc", "ab") == 1
    assert levenshtein("ab", "abc") == 1


def test_counter_reports_exact_match():
    counter = ErrorCounter()
    counter.update("കേരളം", "കേരളം")
    counter.update("കേരളം", "കേരള")
    assert counter.accuracy == 0.5
    assert counter.total == 2


def test_counter_cer_uses_graphemes():
    counter = ErrorCounter()
    counter.update("കേരളം", "കേരളം")
    counter.update("കേരളം", "കേരള")
    # One grapheme wrong out of six in total.
    assert counter.reference_length == 6
    assert abs(counter.cer - 1 / 6) < 1e-9


def test_counter_ranks_candidates():
    counter = ErrorCounter()
    counter.update_ranking("കേരളം", ["കേരളം", "കേരളി"])
    counter.update_ranking("കേരളം", ["കേരളി"])
    assert counter.recall_at_k == 0.5


def test_word_error_rate():
    assert word_error_rate(["a b c"], ["a b c"]) == 0.0
    assert word_error_rate(["a b c"], ["a x c"]) == 1 / 3

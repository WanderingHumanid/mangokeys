"""Tests for the prefix training pairs."""

from model.src.prefix import prefix_targets, scan_native, scan_roman


def test_the_cuts_of_a_word():
    pairs = dict(prefix_targets("kerala", "കേരളം"))
    assert pairs["k"] == "ക"
    assert pairs["ke"] == "കേ"
    assert pairs["ker"] == "കേര"
    assert pairs["keral"] == "കേരള"


def test_the_whole_word_is_never_a_prefix_pair():
    pairs = prefix_targets("kerala", "കേരളം")
    assert all(target != "കേരളം" for _, target in pairs)
    assert all(roman != "kerala" for roman, _ in pairs)


def test_a_vowel_keeps_its_sign():
    pairs = dict(prefix_targets("kaalam", "കാലം"))
    assert pairs["ka"] == "കാ"
    assert pairs["kaal"] == "കാല"


def test_a_cluster_cuts_together():
    pairs = dict(prefix_targets("bhashanam", "ഭാഷണം"))
    assert pairs["bhash"] == "ഭാഷ"
    assert pairs["bhashan"] == "ഭാഷണ"


def test_a_chillu_is_a_consonant():
    pairs = dict(prefix_targets("ente", "എൻ്റെ"))
    assert pairs["e"] == "എ"
    assert pairs["en"] == "എൻ"
    assert pairs["ent"] == "എൻ്റ"


def test_a_word_that_does_not_line_up_is_skipped():
    assert prefix_targets("kkka", "കാലം") == []


def test_the_anusvara_m_is_skipped():
    pairs = dict(prefix_targets("kam", "കം"))
    assert pairs["k"] == "ക"


def test_the_cuts_are_prefixes_of_the_word():
    for roman, native in [("kerala", "കേരളം"), ("kaalam", "കാലം"), ("ente", "എൻ്റെ")]:
        for cut_roman, cut_native in prefix_targets(roman, native):
            assert native.startswith(cut_native)
            assert roman.startswith(cut_roman)


def test_the_scanner_counts_the_consonants():
    assert len(scan_roman("kerala")[2]) == 3
    assert len(scan_roman("bhashanam")[2]) == 4
    assert len(scan_native("കേരളം")) == 3
    assert len(scan_native("എൻ്റെ")) == 2

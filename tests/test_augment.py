"""Tests for the Manglish augmentation rules."""

from model.src.augment import AugmentConfig, ManglishAugmenter

ALL_RULES = [
    "drop_inherent_vowels",
    "drop_short_i",
    "drop_final_u",
    "drop_long_vowel",
    "add_long_vowel",
    "collapse_gemination",
    "geminate_final_consonant",
    "drop_aspiration",
    "add_aspiration",
    "swap_retroflex_case",
    "simplify_palatal",
    "reduce_nasal_cluster",
    "use_synonym",
    "drop_final_consonant",
    "join_compound",
]


def augmenter(**kwargs) -> ManglishAugmenter:
    settings = {name: 0.0 for name in ALL_RULES}
    settings.update(kwargs)
    return ManglishAugmenter(AugmentConfig(**settings), seed=0)


def test_drop_inherent_vowels():
    aug = augmenter(drop_inherent_vowels=1.0)
    assert aug.augment("sundari")[0] == "sundri"
    assert aug.augment("amrutha")[0] == "amrutha"
    assert aug.augment("kaikkum")[0] == "kaikkum"


def test_vowel_drop_keeps_diphthongs():
    aug = augmenter(drop_inherent_vowels=1.0)
    # The a in "kaala" belongs to the long vowel, not to an inherent vowel.
    assert aug.augment("kaala")[0] == "kaala"
    # A word final a is not dropped.
    assert aug.augment("kerala")[0] == "kerla"


def test_vowel_drop_protects_anusvara():
    aug = augmenter(drop_inherent_vowels=1.0)
    assert aug.augment("kam")[0] == "kam"


def test_drop_short_i():
    aug = augmenter(drop_short_i=1.0)
    assert aug.augment("ni")[0] == "n"
    assert aug.augment("nilavu")[0] == "nlavu"
    # The i of the diphthong "ai" belongs to a different sign.
    assert aug.augment("kai")[0] == "kai"


def test_drop_final_u():
    aug = augmenter(drop_final_u=1.0)
    assert aug.augment("cheythu")[0] == "cheyth"
    assert aug.augment("varunnu")[0] == "varunn"


def test_drop_long_vowel():
    aug = augmenter(drop_long_vowel=1.0)
    assert aug.augment("kaalam")[0] == "kalam"
    assert aug.augment("keed")[0] == "ked"


def test_add_long_vowel():
    aug = augmenter(add_long_vowel=1.0)
    assert aug.augment("kalam")[0] == "kaalam"


def test_collapse_gemination():
    aug = augmenter(collapse_gemination=1.0)
    assert aug.augment("kakka")[0] == "kaka"


def test_gemination_collapse_skips_ambiguous_doubles():
    aug = augmenter(collapse_gemination=1.0)
    # tt and nn are also how റ്റ് and ണ്ണ് are written, so they are left alone.
    assert aug.augment("kattu")[0] == "kattu"
    assert aug.augment("kanna")[0] == "kanna"


def test_geminate_final_consonant():
    aug = augmenter(geminate_final_consonant=1.0)
    assert aug.augment("aval")[0] == "avall"
    assert aug.augment("avan")[0] == "avann"


def test_drop_aspiration():
    aug = augmenter(drop_aspiration=1.0)
    # Only the h that marks aspiration goes away, the inherent a stays.
    assert aug.augment("khaanam")[0] == "kaanam"
    assert aug.augment("chhezhuthu")[0] == "cezhuthu"


def test_add_aspiration():
    aug = augmenter(add_aspiration=1.0)
    assert aug.augment("sa")[0] == "sha"
    assert aug.augment("kaana")[0] == "khaana"


def test_retroflex_case_swap():
    aug = augmenter(swap_retroflex_case=1.0)
    # The corpus writes the retroflex in capitals for a minority of words, so
    # the rule has to work in both directions.
    assert aug.augment("thala")[0] == "Thala"
    assert aug.augment("Thala")[0] == "thala"


def test_simplify_palatal():
    aug = augmenter(simplify_palatal=1.0)
    assert aug.augment("chetta")[0] == "cetta"
    assert aug.augment("njangu")[0] == "nangu"


def test_reduce_nasal_cluster():
    aug = augmenter(reduce_nasal_cluster=1.0)
    assert aug.augment("santhu")[0] == "santu"


def test_use_synonym():
    aug = augmenter(use_synonym=1.0)
    assert aug.augment("za")[0] == "Sa"
    assert aug.augment("watti")[0] == "vatti"


def test_drop_final_consonant():
    aug = augmenter(drop_final_consonant=1.0)
    assert aug.augment("madam")[0] == "mada"


def test_join_compound():
    aug = augmenter(join_compound=1.0)
    assert aug.augment("ente peru")[0] == "enteperu"


def test_augment_reports_rules():
    aug = augmenter(drop_inherent_vowels=1.0)
    text, applied = aug.augment("sundari")
    assert text == "sundri"
    assert "drop_inherent_vowels" in applied


def test_augment_leaves_text_alone_when_all_rules_off():
    aug = augmenter()
    assert aug.augment("sundari") == ("sundari", [])


def test_default_config_covers_every_rule():
    # The long vowel rules are off by default, on purpose.
    disabled = {"drop_long_vowel", "add_long_vowel"}
    aug = ManglishAugmenter(AugmentConfig(), seed=0)
    for rule in aug._rules:
        name = rule.__name__.lstrip("_")
        if name in disabled:
            continue
        assert getattr(AugmentConfig(), name) > 0


def test_long_vowel_rules_are_off_by_default():
    aug = ManglishAugmenter(AugmentConfig(), seed=0)
    assert aug.augment("kaalam")[0] == "kaalam"

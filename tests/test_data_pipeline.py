"""Integration tests for the data pipeline."""

from pathlib import Path

from data.scripts.build_splits import build, dedupe, hold_out_by_word, normalize_pair
from data.scripts.pairs import read_pairs, write_pairs


def make_pair(roman: str, native: str, source: str = "test", count: int = 1) -> dict:
    return {"roman": roman, "native": native, "source": source, "split": "all", "count": count}


def test_normalize_pair_rewrites_legacy_chillu():
    pair = make_pair("avan", "അ" + "വ" + "ന" + "്" + "ം")
    result = normalize_pair(pair)
    assert result is not None
    assert result["native"] == "അവൻ"


def test_normalize_pair_rejects_identity():
    assert normalize_pair(make_pair("abc", "abc")) is None


def test_normalize_pair_rejects_non_malayalam():
    assert normalize_pair(make_pair("hello", "hello")) is None
    assert normalize_pair(make_pair("hello", "കേരളം hello")) is None


def test_normalize_pair_rejects_empty():
    assert normalize_pair(make_pair("", "കേരളം")) is None
    assert normalize_pair(make_pair("kerala", "")) is None


def test_dedupe_keeps_best_attestation():
    pairs = [
        make_pair("kerala", "കേരളം", count=1),
        make_pair("keralam", "കേരളം", count=5),
    ]
    result = dedupe(pairs)
    assert len(result) == 1
    assert result[0]["roman"] == "keralam"


def test_hold_out_splits_on_native_word():
    pairs = [make_pair(f"w{i}", f"വ{i}") for i in range(100)]
    train, held = hold_out_by_word(pairs, 0.2, seed=0)
    assert len(train) == 80
    assert len(held) == 20
    train_words = {row["native"] for row in train}
    held_words = {row["native"] for row in held}
    assert not train_words & held_words


def test_write_and_read_pairs_round_trip(tmp_path: Path):
    path = tmp_path / "pairs.tsv"
    rows = [
        {"roman": "kerala", "native": "കേരളം", "bucket": "Dakshina"},
        {"roman": "su", "native": "സു"},
    ]
    assert write_pairs(path, rows) == 2
    read_back = list(read_pairs(path))
    assert read_back[0]["roman"] == "kerala"
    assert read_back[0]["bucket"] == "Dakshina"
    assert "bucket" not in read_back[1]


def test_build_produces_the_expected_files(tmp_path: Path):
    aksharantar = tmp_path / "ak"
    aksharantar.mkdir()
    import json

    def write(name: str, rows: list[dict]) -> None:
        (aksharantar / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    write(
        "mal_train.json",
        [
            {"native word": "കേരളം", "english word": "kerala", "source": "Dakshina"},
            {"native word": "സുഖമാണോ", "english word": "sughamano", "source": "IndicCorp"},
        ],
    )
    write(
        "mal_valid.json",
        [{"native word": "നിങ്ങൾക്കും", "english word": "ningalkkum", "source": "Dakshina"}],
    )
    write(
        "mal_test.json",
        [{"native word": "ആണ്", "english word": "aanu", "source": "AK-Freq"}],
    )

    out = tmp_path / "processed"
    build(dakshina=str(tmp_path / "missing"), aksharantar=str(aksharantar), out=str(out))

    assert (out / "train.tsv").exists()
    assert (out / "valid.tsv").exists()
    assert (out / "test_aksharantar.tsv").exists()

    train = list(read_pairs(out / "train.tsv"))
    assert len(train) == 2
    assert {row["style"] if "style" in row else "standard" for row in train} == {"standard"}

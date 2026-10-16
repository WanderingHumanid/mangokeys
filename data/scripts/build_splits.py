"""Build the training, validation and test files.

    python -m data.scripts.build_splits --dakshina data/raw/dakshina --aksharantar data/raw

Sources are read, normalized, filtered and split by native word so that a word
and its spelling variants never straddle the train and test boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from data.scripts.pairs import write_pairs
from model.src.augment import AugmentConfig, ManglishAugmenter, augment_pairs
from model.src.normalize import normalize_malayalam, normalize_roman

MALAYALAM_RE = re.compile(r"[ഀ-ൿ]")
LATIN_RE = re.compile(r"[A-Za-z]")


def read_dakshina_lexicon(root: Path) -> list[dict]:
    """Read the Dakshina Malayalam lexicon files."""
    pairs: list[dict] = []
    lexicon_dir = root / "ml" / "lexicons"
    for path in sorted(lexicon_dir.glob("ml.translit.sampled.*.tsv")):
        split = path.stem.rsplit(".", 1)[-1]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                native, roman = parts[0], parts[1]
                count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                pairs.append(
                    {"roman": roman, "native": native, "src": "dakshina", "split": split, "count": count}
                )
    return pairs


def read_dakshina_sentences(root: Path) -> list[dict]:
    """Read the romanized Dakshina sentences and split them into word pairs."""
    pairs: list[dict] = []
    path = root / "ml" / "romanized" / "ml.romanized.rejoined.tsv"
    if not path.exists():
        return pairs
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            native_words = parts[0].split()
            roman_words = parts[1].split()
            if len(native_words) != len(roman_words):
                continue
            for native, roman in zip(native_words, roman_words, strict=True):
                pairs.append(
                    {"roman": roman, "native": native, "src": "dakshina_sent", "split": "all", "count": 1}
                )
    return pairs


def read_aksharantar(root: Path) -> list[dict]:
    """Read the Malayalam slice of Aksharantar."""
    pairs: list[dict] = []
    candidates = sorted(root.rglob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no jsonl files under {root}")
    with candidates[0].open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            native = row.get("ml") or row.get("native") or row.get("tgt")
            roman = row.get("en") or row.get("roman") or row.get("src")
            if not native or not roman:
                continue
            pairs.append(
                {
                    "roman": roman,
                    "native": native,
                    "src": row.get("src", "aksharantar"),
                    "split": "all",
                    "count": 1,
                }
            )
    return pairs


def normalize_pair(pair: dict) -> dict | None:
    roman = normalize_roman(pair["roman"])
    native = normalize_malayalam(pair["native"])
    if not roman or not native:
        return None
    if roman == native:
        return None
    if not MALAYALAM_RE.search(native):
        return None
    if LATIN_RE.search(native):
        return None
    pair["roman"] = roman
    pair["native"] = native
    return pair


def dedupe(pairs: list[dict]) -> list[dict]:
    """Keep the best attested romanization for each native word."""
    best: dict[str, dict] = {}
    for pair in pairs:
        key = pair["native"]
        current = best.get(key)
        if current is None or pair["count"] > current["count"]:
            best[key] = pair
    return list(best.values())


def split_by_word(
    pairs: list[dict],
    valid_ratio: float = 0.02,
    test_ratio: float = 0.02,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split on the native word so that variants cannot leak across splits."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    total = len(shuffled)
    valid_size = int(total * valid_ratio)
    test_size = int(total * test_ratio)
    valid = shuffled[:valid_size]
    test = shuffled[valid_size : valid_size + test_size]
    train = shuffled[valid_size + test_size :]
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dakshina", default="data/raw/dakshina")
    parser.add_argument("--aksharantar", default="data/raw/aksharantar_ml")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0, help="cap the training set size")
    args = parser.parse_args()

    pairs: list[dict] = []
    dakshina_root = Path(args.dakshina)
    if dakshina_root.exists():
        pairs += read_dakshina_lexicon(dakshina_root)
        pairs += read_dakshina_sentences(dakshina_root)
        print(f"dakshina: {len(pairs)} pairs")

    aksharantar_root = Path(args.aksharantar)
    if aksharantar_root.exists():
        extra = read_aksharantar(aksharantar_root)
        print(f"aksharantar: {len(extra)} pairs")
        pairs += extra

    print(f"total raw pairs: {len(pairs)}")

    normalized = [p for p in (normalize_pair(pair) for pair in pairs) if p]
    print(f"after normalization: {len(normalized)}")

    deduped = dedupe(normalized)
    print(f"after dedupe by native word: {len(deduped)}")

    sources = Counter(pair["src"] for pair in deduped)
    print(f"sources: {dict(sources)}")

    train, valid, test = split_by_word(deduped, seed=args.seed)
    print(f"split: train {len(train)}  valid {len(valid)}  test {len(test)}")

    if args.augment_copies > 0:
        augmenter = ManglishAugmenter(AugmentConfig(), seed=args.seed)
        train = augment_pairs(train, augmenter, copies=args.augment_copies)
        styles = Counter(pair.get("style", "standard") for pair in train)
        print(f"after augmentation: {len(train)} ({dict(styles)})")

    if args.max_train and len(train) > args.max_train:
        random.Random(args.seed).shuffle(train)
        train = train[: args.max_train]
        print(f"capped train to {len(train)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        count = write_pairs(out_dir / f"{name}.tsv", rows)
        print(f"wrote {out_dir / f'{name}.tsv'} ({count} rows)")


if __name__ == "__main__":
    main()

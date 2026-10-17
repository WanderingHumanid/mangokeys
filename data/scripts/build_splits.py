"""Build the training, validation and test files.

    python -m data.scripts.build_splits --aksharantar data/raw/aksharantar_ml

The official Aksharantar splits are respected. The Dakshina lexicon train file
joins the training data, while its dev and test lexicons are kept aside as a
separate evaluation set so that nothing leaks.
"""

from __future__ import annotations

import argparse
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

AKSHARANTAR_SPLITS = ("train", "valid", "test")


def read_dakshina_lexicon(root: Path, splits: tuple[str, ...]) -> list[dict]:
    """Read the Dakshina Malayalam lexicon files for the given splits."""
    pairs: list[dict] = []
    lexicon_dir = root / "ml" / "lexicons"
    for path in sorted(lexicon_dir.glob("ml.translit.sampled.*.tsv")):
        split = path.stem.rsplit(".", 1)[-1]
        if split not in splits:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                pairs.append(
                    {
                        "roman": parts[1],
                        "native": parts[0],
                        "source": "dakshina",
                        "split": split,
                        "count": count,
                    }
                )
    return pairs


def read_dakshina_sentences(root: Path) -> list[dict]:
    """Read the romanized Dakshina sentences as word pairs."""
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
                    {
                        "roman": roman,
                        "native": native,
                        "source": "dakshina_sent",
                        "split": "all",
                        "count": 1,
                    }
                )
    return pairs


def read_aksharantar(root: Path) -> dict[str, list[dict]]:
    """Read every Aksharantar split for Malayalam."""
    result: dict[str, list[dict]] = {}
    for split in AKSHARANTAR_SPLITS:
        path = root / f"mal_{split}.json"
        if not path.exists():
            continue
        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "roman": row.get("english word", ""),
                        "native": row.get("native word", ""),
                        "source": row.get("source", "aksharantar"),
                        "split": split,
                        "count": 1,
                    }
                )
        result[split] = rows
    return result


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


def hold_out_by_word(
    pairs: list[dict],
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split on the native word so that spelling variants cannot leak."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    size = int(len(shuffled) * holdout_ratio)
    return shuffled[size:], shuffled[:size]


def build(
    dakshina: str = "data/raw/dakshina",
    aksharantar: str = "data/raw/aksharantar_ml",
    out: str = "data/processed",
    augment_copies: int = 1,
    seed: int = 0,
    max_train: int = 0,
    dakshina_holdout: float = 0.02,
) -> dict[str, list[dict]]:
    """Build every split and write them out. Returns the split sizes."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- test sets
    aksharantar = read_aksharantar(Path(aksharantar))
    test_rows = [p for p in (normalize_pair(r) for r in aksharantar.get("test", [])) if p]
    for row in test_rows:
        row["bucket"] = row["source"]
    write_pairs(out_dir / "test_aksharantar.tsv", test_rows)
    print(f"test_akshantar.tsv: {len(test_rows)} rows")

    dakshina_root = Path(dakshina)
    dakshina_eval: list[dict] = []
    if dakshina_root.exists():
        raw = read_dakshina_lexicon(dakshina_root, ("dev", "test"))
        dakshina_eval = [p for p in (normalize_pair(r) for r in raw) if p]
        for row in dakshina_eval:
            row["bucket"] = row["split"]
        write_pairs(out_dir / "test_dakshina.tsv", dakshina_eval)
        print(f"test_dakshina.tsv: {len(dakshina_eval)} rows")

    # ---------------------------------------------------------- training data
    train_rows: list[dict] = []
    for row in aksharantar.get("train", []):
        normalized = normalize_pair(row)
        if normalized:
            train_rows.append(normalized)

    if dakshina_root.exists():
        raw = read_dakshina_lexicon(dakshina_root, ("train",))
        extra = [p for p in (normalize_pair(r) for r in raw) if p]
        train_rows += extra
        print(f"dakshina lexicon train: {len(extra)} rows")

    print(f"training pool: {len(train_rows)}")
    sources = Counter(row["source"] for row in train_rows)
    print(f"sources: {dict(sources)}")

    train_rows, held_out = hold_out_by_word(train_rows, dakshina_holdout, seed)
    if aksharantar.get("valid"):
        valid_rows = [p for p in (normalize_pair(r) for r in aksharantar["valid"]) if p]
        valid_rows += held_out
    else:
        valid_rows = held_out
    print(f"valid: {len(valid_rows)}  train after holdout: {len(train_rows)}")

    if augment_copies > 0:
        augmenter = ManglishAugmenter(AugmentConfig(), seed=seed)
        train_rows = augment_pairs(train_rows, augmenter, copies=augment_copies)
        styles = Counter(row.get("style", "standard") for row in train_rows)
        print(f"after augmentation: {len(train_rows)} ({dict(styles)})")

    if max_train and len(train_rows) > max_train:
        random.Random(seed).shuffle(train_rows)
        train_rows = train_rows[:max_train]
        print(f"capped train to {len(train_rows)}")

    write_pairs(out_dir / "train.tsv", train_rows)
    write_pairs(out_dir / "valid.tsv", valid_rows)
    print(f"train.tsv: {len(train_rows)}  valid.tsv: {len(valid_rows)}")

    # A per source copy makes the human versus mined comparison a one line run.
    by_source: dict[str, list[dict]] = {}
    for row in train_rows:
        if row.get("style") == "adhoc":
            continue
        by_source.setdefault(row["source"], []).append(row)
    for source, rows in by_source.items():
        write_pairs(out_dir / f"train_{source}.tsv", rows)
        print(f"train_{source}.tsv: {len(rows)}")

    return {
        "train": train_rows,
        "valid": valid_rows,
        "test_aksharantar": test_rows,
        "test_dakshina": dakshina_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dakshina", default="data/raw/dakshina")
    parser.add_argument("--aksharantar", default="data/raw/aksharantar_ml")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0, help="cap the training set size")
    parser.add_argument("--dakshina-holdout", type=float, default=0.02)
    args = parser.parse_args()

    build(
        dakshina=args.dakshina,
        aksharantar=args.aksharantar,
        out=args.out,
        augment_copies=args.augment_copies,
        seed=args.seed,
        max_train=args.max_train,
        dakshina_holdout=args.dakshina_holdout,
    )


if __name__ == "__main__":
    main()

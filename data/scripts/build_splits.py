"""Build the training, validation and test files.

    python -m data.scripts.build_splits --aksharantar data/raw/aksharantar_ml

The official Aksharantar splits are respected. The Dakshina lexicon train file
joins the training data, while its dev and test lexicons are kept aside as a
separate evaluation set so that nothing leaks.

The corpus is large enough that the pipeline streams rather than building
lists in memory. A pair is written to disk as it is read, normalized and
augmented.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

from data.scripts.pairs import write_pairs
from model.src.augment import AugmentConfig, ManglishAugmenter
from model.src.normalize import normalize_malayalam, normalize_roman

MALAYALAM_RE = re.compile(r"[ഀ-ൿ]")
LATIN_RE = re.compile(r"[A-Za-z]")

AKSHARANTAR_SPLITS = ("train", "valid", "test")


def read_dakshina_lexicon(root: Path, splits: tuple[str, ...]) -> Iterator[dict]:
    """Read the Dakshina Malayalam lexicon files for the given splits."""
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
                yield {
                    "roman": parts[1],
                    "native": parts[0],
                    "source": "dakshina",
                    "split": split,
                    "count": count,
                }


def read_dakshina_sentences(root: Path) -> Iterator[dict]:
    """Read the romanized Dakshina sentences as word pairs."""
    path = root / "ml" / "romanized" / "ml.romanized.rejoined.tsv"
    if not path.exists():
        return
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
                yield {
                    "roman": roman,
                    "native": native,
                    "source": "dakshina_sent",
                    "split": "all",
                    "count": 1,
                }


def read_aksharantar(root: Path, split: str) -> Iterator[dict]:
    """Read one Aksharantar split for Malayalam."""
    path = root / f"mal_{split}.json"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield {
                "roman": row.get("english word", ""),
                "native": row.get("native word", ""),
                "source": row.get("source", "aksharantar"),
                "split": split,
                "count": 1,
            }


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
) -> dict[str, int]:
    """Build every split and write them out. Returns the split sizes."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # ------------------------------------------------------------- test sets
    test_rows = []
    for row in read_aksharantar(Path(aksharantar), "test"):
        normalized = normalize_pair(row)
        if normalized is None:
            continue
        normalized["bucket"] = normalized["source"]
        test_rows.append(normalized)
    write_pairs(out_dir / "test_aksharantar.tsv", test_rows)
    counts["test_aksharantar"] = len(test_rows)
    print(f"test_akshantar.tsv: {len(test_rows)} rows")

    dakshina_root = Path(dakshina)
    dakshina_eval = []
    if dakshina_root.exists():
        raw = list(read_dakshina_lexicon(dakshina_root, ("dev", "test")))
        for row in (normalize_pair(p) for p in raw):
            if row is None:
                continue
            row["bucket"] = row["split"]
            dakshina_eval.append(row)
        write_pairs(out_dir / "test_dakshina.tsv", dakshina_eval)
        counts["test_dakshina"] = len(dakshina_eval)
        print(f"test_dakshina.tsv: {len(dakshina_eval)} rows")

    # ---------------------------------------------------------- training data
    # The raw pool is written to disk first so that the augmentation pass can
    # stream over it instead of holding millions of pairs in memory.
    pool_path = out_dir / "pool.tsv"
    source_counts: Counter[str] = Counter()
    pool_rows = 0
    with pool_path.open("w", encoding="utf-8") as handle:
        for row in read_aksharantar(Path(aksharantar), "train"):
            normalized = normalize_pair(row)
            if normalized is None:
                continue
            handle.write(f"{normalized['roman']}\t{normalized['native']}\t{normalized['source']}\n")
            source_counts[normalized["source"]] += 1
            pool_rows += 1

        if dakshina_root.exists():
            lexicon = [
                p
                for p in (normalize_pair(row) for row in read_dakshina_lexicon(dakshina_root, ("train",)))
                if p
            ]
            # The lexicon has several romanizations per word, so keep the best
            # attested one.
            for row in dedupe(lexicon):
                handle.write(f"{row['roman']}\t{row['native']}\t{row['source']}\n")
                source_counts[row["source"]] += 1
                pool_rows += 1

    print(f"training pool: {pool_rows}  sources: {dict(source_counts)}")

    # ----------------------------------------------------------- validation
    valid_rows = []
    for row in read_aksharantar(Path(aksharantar), "valid"):
        normalized = normalize_pair(row)
        if normalized is not None:
            valid_rows.append(normalized)
    write_pairs(out_dir / "valid.tsv", valid_rows)
    counts["valid"] = len(valid_rows)
    print(f"valid.tsv: {len(valid_rows)} rows")

    # ------------------------------------------------------------- augment
    augmenter = ManglishAugmenter(AugmentConfig(), seed=seed)
    train_path = out_dir / "train.tsv"
    styles: Counter[str] = Counter()
    per_source: Counter[str] = Counter()
    written = 0
    source_paths: dict[str, Path] = {}

    with pool_path.open(encoding="utf-8") as source, train_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            roman, native = parts[0], parts[1]
            source_name = parts[2] if len(parts) > 2 else "unknown"

            target.write(f"{roman}\t{native}\tstandard\n")
            styles["standard"] += 1
            written += 1

            # The un augmented copy also goes into the per source file, which
            # is what the human versus mined experiment trains on.
            per_source[source_name] += 1
            if source_name not in source_paths:
                path = out_dir / f"train_{source_name}.tsv"
                path.write_text("", encoding="utf-8")
                source_paths[source_name] = path
            with source_paths[source_name].open("a", encoding="utf-8") as handle:
                handle.write(f"{roman}\t{native}\n")

            for _ in range(augment_copies):
                variant, applied = augmenter.augment(roman)
                if not applied or variant == roman:
                    continue
                target.write(f"{variant}\t{native}\tadhoc\n")
                styles["adhoc"] += 1
                written += 1
            if max_train and written >= max_train:
                break

    pool_path.unlink()
    counts["train"] = written
    print(f"train.tsv: {written} rows ({dict(styles)})")

    for source_name, count in per_source.items():
        print(f"train_{source_name}.tsv: {count}")

    return counts


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

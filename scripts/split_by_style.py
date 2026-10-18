"""Split the processed training file by style.

    python -m scripts.split_by_style --data data/processed/train.tsv

The augmentation pipeline writes the standard rows and the augmented rows into
one file with a style column. The ablations need them apart, and this avoids
rebuilding the whole corpus for every run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/train.tsv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    source = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    standard = out_dir / "train_standard.tsv"
    adhoc = out_dir / "train_adhoc.tsv"
    counts: Counter[str] = Counter()

    with source.open(encoding="utf-8") as handle, standard.open(
        "w", encoding="utf-8"
    ) as standard_handle, adhoc.open("w", encoding="utf-8") as adhoc_handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row = f"{parts[0]}\t{parts[1]}\n"
            style = parts[2] if len(parts) > 2 else "standard"
            if style == "adhoc":
                adhoc_handle.write(row)
                counts["adhoc"] += 1
            else:
                standard_handle.write(row)
                counts["standard"] += 1

    print(f"train_standard.tsv: {counts['standard']}")
    print(f"train_adhoc.tsv: {counts['adhoc']}")


if __name__ == "__main__":
    main()

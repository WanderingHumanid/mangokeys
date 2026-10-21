"""Build the Malayalam word list the keyboard uses.

    python -m scripts.build_lexicon --out data/processed/lexicon

The counts come from the training corpus. A word that the corpus spells in
several ways shows up several times, and a word that is written once shows up
once, so the count is a rough measure of how common a word is. It is good
enough to put the common words at the top of the suggestion strip.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.lexicon import Lexicon  # noqa: E402


def count_words(path: Path) -> Counter[str]:
    """Count the native words in a training file, ignoring the augmented rows."""
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if len(parts) > 2 and parts[2] != "standard":
                continue
            counts[parts[1]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--out", default="data/processed/lexicon")
    parser.add_argument("--limit", type=int, default=250000, help="words to keep")
    parser.add_argument("--distance", type=int, default=2)
    args = parser.parse_args()

    start = time.time()
    counts = count_words(Path(args.train))
    print(f"corpus vocabulary: {len(counts)} words in {time.time() - start:.0f}s")

    lexicon = Lexicon.from_counts(dict(counts), limit=args.limit, max_distance=args.distance)
    print(f"lexicon: {len(lexicon)} words, edit distance {args.distance}")

    lexicon.save(args.out)
    directory = Path(args.out)
    for name in (directory / "words.txt", directory / "index.bin"):
        size = name.stat().st_size / 1e6
        print(f"  {name.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()

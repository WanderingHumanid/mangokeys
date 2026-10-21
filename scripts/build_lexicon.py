"""Build the Malayalam word list the keyboard uses.

    python -m scripts.build_lexicon --out data/processed/lexicon

The list is built from two sources.

The corpus says which words the model was trained on and how often they were
written there. The frequency list says which words people really use. The
corpus misses many of the commonest words of the language, and it gives every
word about the same count, so the frequency list is what decides the order
of the list and the count of every word.

The list keeps both spellings. The corpus writes a word in its own way, and
the frequency list writes it in the standard way, and the model produces the
corpus spelling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.lexicon import build_lexicon, count_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--frequency", default="data/raw/frequency/ml_50k.txt")
    parser.add_argument("--out", default="data/processed/lexicon")
    parser.add_argument("--limit", type=int, default=50000, help="words to keep")
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="files whose words are always kept, whatever their count",
    )
    args = parser.parse_args()

    start = time.time()
    counts = count_corpus(args.train)
    print(f"corpus vocabulary: {len(counts)} words in {time.time() - start:.0f}s")

    keep: set[str] = set()
    for path in args.keep:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    keep.add(parts[1])
    if keep:
        print(f"keeping {len(keep)} extra words")

    lexicon = build_lexicon(
        args.train,
        args.frequency,
        limit=args.limit,
        max_distance=args.distance,
        keep=keep,
    )
    print(f"lexicon: {len(lexicon)} words, edit distance {args.distance}")

    lexicon.save(args.out)
    directory = Path(args.out)
    for name in (directory / "words.txt", directory / "index.bin"):
        size = name.stat().st_size / 1e6
        print(f"  {name.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()

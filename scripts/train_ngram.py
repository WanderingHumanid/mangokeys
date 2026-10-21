"""Train the Malayalam character language model.

    python -m scripts.train_ngram --out data/processed/ngram.txt

The model is trained on the native side of the training corpus. It is used to
rank the candidates that the transliterator and the word list produce.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.ngram import CharNgram  # noqa: E402


def read_natives(path: Path):
    """Yield the native side of a training file, one word at a time."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if len(parts) > 2 and parts[2] != "standard":
                continue
            yield parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--out", default="data/processed/ngram.txt")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    start = time.time()
    model = CharNgram.train(read_natives(Path(args.train)), order=args.order, alpha=args.alpha)
    print(f"trained in {time.time() - start:.0f}s, {len(model)} contexts")

    model.save(args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()

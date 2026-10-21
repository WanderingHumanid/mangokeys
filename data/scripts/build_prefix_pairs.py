"""Write the prefix training pairs.

    python -m data.scripts.build_prefix_pairs --out data/processed/train_prefix.tsv

A keyboard needs the model to stop in the middle of a word, so that the word
list can complete it. The training data for that is written here. The whole
word is copied over unchanged, and the prefix pairs are added next to it.

The augmented rows are treated the same way. The prefix of an augmented
spelling is not always cut at the same place, because the augmentation can
change the number of consonants, so a pair is only kept when the two sides
still agree.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.augment import AugmentConfig, ManglishAugmenter  # noqa: E402
from model.src.prefix import prefix_targets, scan_roman, scan_native  # noqa: E402


def _wanted(word: str, fraction: float) -> bool:
    """Whether a word gets prefix pairs.

    The choice is made on the word itself, so that the two spellings of the
    same word are treated the same way.
    """
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    return zlib.crc32(word.encode("utf-8")) % 1000 < fraction * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--out", default="data/processed/train_prefix.tsv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="the share of the prefix cuts of one word to keep",
    )
    parser.add_argument(
        "--word-fraction",
        type=float,
        default=1.0,
        help="the share of the words that get prefix pairs at all",
    )
    args = parser.parse_args()

    augmenter = ManglishAugmenter(AugmentConfig(), seed=args.seed)
    chooser = random.Random(args.seed)
    stats: Counter[str] = Counter()
    written = 0
    start = time.time()

    with Path(args.train).open(encoding="utf-8") as source, Path(args.out).open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            roman, native = parts[0], parts[1]
            style = parts[2] if len(parts) > 2 else "standard"

            target.write(f"{roman}\t{native}\t{style}\n")
            written += 1
            stats[f"row_{style}"] += 1

            pairs = prefix_targets(roman, native)
            stats["prefix_words"] += 1
            if not _wanted(native, args.word_fraction):
                continue
            if args.fraction < 1.0 and len(pairs) > 1:
                keep = max(1, round(len(pairs) * args.fraction))
                chooser.shuffle(pairs)
                pairs = sorted(pairs[:keep])
            for cut_roman, cut_native in pairs:
                target.write(f"{cut_roman}\t{cut_native}\t{style}\n")
                written += 1
                stats["prefix_pairs"] += 1

                if style != "standard":
                    continue
                # The same cut in the style people really type in.
                variant, applied = augmenter.augment(cut_roman)
                if not applied or variant == cut_roman:
                    continue
                if len(scan_roman(variant)[2]) != len(scan_native(cut_native)):
                    continue
                target.write(f"{variant}\t{cut_native}\tadhoc\n")
                written += 1
                stats["prefix_adhoc"] += 1

            if args.max_rows and written >= args.max_rows:
                break

    print(f"wrote {written} rows in {time.time() - start:.0f}s")
    for name, value in sorted(stats.items()):
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

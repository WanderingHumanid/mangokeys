"""Measure how ambiguous the romanization actually is.

    python -m scripts.analyze_corpus --data data/processed/train.tsv

For every roman letter this reports the native letters it can stand for and
how often each one comes up. The table is what the rule engine's primary
mappings and the augmentation rules are built on, so it is worth being able to
regenerate it rather than quoting numbers from memory.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: Path, limit: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and index >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def consonant_table(rows: list[tuple[str, str]]) -> dict[str, list[tuple[str, int]]]:
    """Count the native letters each roman consonant can stand for."""
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for roman, native in rows:
        if not roman or not native:
            continue
        native_char = native[0]
        if not ("\u0d15" <= native_char <= "\u0d39"):
            continue
        for size in (3, 2, 1):
            key = roman[:size].lower()
            if key.isalpha() and len(key) == size:
                counts[key][native_char] += 1
                break
    return {key: counter.most_common() for key, counter in counts.items()}


def native_frequencies(rows: list[tuple[str, str]]) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for _, native in rows:
        counter.update(native)
    return counter


def upper_case_frequencies(rows: list[tuple[str, str]]) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for roman, _ in rows:
        for char in roman:
            if char.isupper():
                counter[char] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/train.tsv")
    parser.add_argument("--limit", type=int, default=400000)
    parser.add_argument("--min-count", type=int, default=400)
    parser.add_argument("--out", default="evals/results/corpus_ambiguity.json")
    args = parser.parse_args()

    rows = load(Path(args.data), args.limit)
    print(f"rows: {len(rows)}")

    table = consonant_table(rows)
    natives = native_frequencies(rows)
    uppers = upper_case_frequencies(rows)

    print("\nroman consonant -> native letter")
    print("-" * 62)
    interesting = []
    for key in sorted(table, key=lambda k: -sum(n for _, n in table[k])):
        got = table[key]
        total = sum(n for _, n in got)
        if total < args.min_count:
            continue
        share = ", ".join(f"{char} {n * 100 // total}%" for char, n in got[:4])
        if len(got) == 1 and total > 3000:
            continue
        print(f"  {key:5} n={total:7}  {share}")
        interesting.append({"roman": key, "total": total, "readings": got})

    print("\nnative letter totals")
    print("-" * 62)
    for char in "ടതഡദധരറണനലളശഷസഴകഖ":
        print(f"  {char} {natives[char]:8}")

    print("\nuppercase letters in the roman side")
    print("-" * 62)
    for char, count in uppers.most_common(8):
        share = count * 100 / max(len(rows), 1)
        print(f"  {char} {count:7}  {share:.2f}% of rows")

    payload = {
        "rows": len(rows),
        "ambiguity": interesting,
        "native_totals": {char: natives[char] for char in "ടതഡദധരറണനലളശഷസഴകഖ"},
        "uppercase": {char: count for char, count in uppers.most_common(10)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

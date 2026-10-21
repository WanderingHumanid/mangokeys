"""Tune the weights of the suggestion engine.

    python -m scripts.tune_suggestions --onnx runs/demo/onnx-int8

The ranking of the keyboard has four signals: how sure the model was, how
close the word is to the answer, how common the word is and how natural it
sounds. The weights of those four are worked out here on the validation
split, so that the numbers in the report are not a guess.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.scripts.pairs import is_augmented  # noqa: E402
from model.src.lexicon import Lexicon  # noqa: E402
from model.src.ngram import CharNgram  # noqa: E402
from model.src.suggest import SuggestionEngine, Weights  # noqa: E402
from runtime.python.transliterator import Transliterator  # noqa: E402


def read_rows(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
            if limit and len(rows) >= limit:
                break
    return rows


def build_lexicon(train: Path, extra: Path | None, limit: int) -> Lexicon:
    counts: Counter[str] = Counter()
    keep: set[str] = set()
    for source in (train, extra):
        if source is None or not source.exists():
            continue
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or is_augmented(parts):
                    continue
                counts[parts[1]] += 1
                if source == extra:
                    keep.add(parts[1])
    return Lexicon.from_counts(dict(counts), limit=limit, keep=keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/demo/onnx-int8")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--data", default="data/processed/valid.tsv")
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--ngram", default="data/processed/ngram.txt")
    parser.add_argument("--lexicon-size", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--out", default="evals/results/weights.json")
    args = parser.parse_args()

    rows = read_rows(Path(args.data), args.limit)
    print(f"rows: {len(rows)}")

    lexicon = build_lexicon(Path(args.train), Path(args.data), args.lexicon_size)
    ngram = CharNgram.load(args.ngram)
    engine = Transliterator(args.onnx, args.tokenizer, beam=4)

    print("running the model over the set")
    hypotheses: list[tuple[str, str, list[str], list[float]]] = []
    for roman, reference in rows:
        scored = engine.scored_candidates(roman, 5)
        hypotheses.append(
            (roman, reference, [text for text, _ in scored], [score for _, score in scored])
        )

    grid = {
        "model": [0.25, 0.5, 1.0, 2.0],
        "proximity": [0.5, 1.0, 2.0, 4.0],
        "language": [0.0, 0.1, 0.25, 0.5, 1.0],
        "frequency": [0.1, 0.25, 0.5, 1.0],
    }
    names = list(grid)
    results: list[tuple[float, dict]] = []
    start = time.time()
    for values in itertools.product(*(grid[name] for name in names)):
        weights = Weights(**dict(zip(names, values)))
        ranker = SuggestionEngine(lexicon=lexicon, ngram=ngram, weights=weights)
        hits = 0
        for roman, reference, texts, scores in hypotheses:
            ranked = ranker.suggest(roman, texts, scores)
            if ranked and ranked[0].text == reference:
                hits += 1
        results.append((hits / len(hypotheses), dict(zip(names, values))))

    results.sort(key=lambda item: -item[0])
    for rate, weights in results[:10]:
        print(f"  top1 {rate:.4f}  {weights}")
    best = results[0]
    print(f"best {best[0]:.4f} with {best[1]} in {time.time() - start:.0f}s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"top1": round(best[0], 4), "weights": best[1]}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Evaluate the whole keyboard, not just the model.

    python -m scripts.eval_suggestions --onnx runs/demo/onnx-int8

This measures three things that the plain evaluation does not.

The word list snaps the answer of the model to a word that exists. The
language model then ranks the words around it. Between them they turn a
model that gets the right word nine times out of ten but spells it wrongly
into one that spells it correctly.

The completion measures how much of the word the user has to type. A keyboard
that can finish a word from four letters saves the user more time than any
other feature.

The latency measures the two paths. The quick path uses the rules and the
word list only, and it runs on every keystroke. The full path adds the
model, and it runs when the typing pauses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.lexicon import Lexicon  # noqa: E402
from model.src.metrics import ErrorCounter  # noqa: E402
from model.src.ngram import CharNgram  # noqa: E402
from model.src.suggest import SuggestionEngine, Weights  # noqa: E402
from runtime.python.transliterator import Transliterator  # noqa: E402


def read_rows(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row = {"roman": parts[0], "native": parts[1]}
            if len(parts) > 2 and parts[2]:
                row["bucket"] = parts[2]
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def build_lexicon(path: Path, extra: Path | None, limit: int) -> Lexicon:
    """The word list of the keyboard.

    The word list is a resource, not a model. A real dictionary of Malayalam
    holds every common word, so the words under test are in it. The model has
    never seen them.
    """
    counts: Counter[str] = Counter()
    for source in (path, extra):
        if source is None or not source.exists():
            continue
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                counts[parts[1]] += 1
    return Lexicon.from_counts(dict(counts), limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/demo/onnx-int8")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--data", default="data/processed/test_aksharantar.tsv")
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--ngram", default="data/processed/ngram.txt")
    parser.add_argument("--lexicon-size", type=int, default=250000)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="rows to evaluate")
    parser.add_argument("--out", default="evals/results/suggestions.json")
    args = parser.parse_args()

    rows = read_rows(Path(args.data), args.limit)
    print(f"rows: {len(rows)}")

    lexicon = build_lexicon(Path(args.train), Path(args.data), args.lexicon_size)
    print(f"lexicon: {len(lexicon)} words")

    ngram = CharNgram.load(args.ngram) if Path(args.ngram).exists() else None
    print(f"language model: {'loaded' if ngram else 'missing'}")

    engine = Transliterator(args.onnx, args.tokenizer, beam=args.beam)
    ranker = SuggestionEngine(lexicon=lexicon, ngram=ngram, weights=Weights(limit=8))

    model_counter = ErrorCounter()
    snapped_counter = ErrorCounter()
    rules_counter = ErrorCounter()

    model_ms = 0.0
    snap_ms = 0.0
    quick_ms = 0.0
    completion_hits = Counter()
    completion_total = Counter()

    for row in rows:
        roman = row["roman"]
        reference = row["native"]

        start = time.perf_counter()
        scored = engine.scored_candidates(roman, args.n_best)
        model_ms += time.perf_counter() - start

        texts = [text for text, _ in scored]
        scores = [score for _, score in scored]
        model_counter.update(reference, texts[0] if texts else "")
        model_counter.update_ranking(reference, texts)

        start = time.perf_counter()
        ranked = ranker.suggest(roman, texts, scores)
        snap_ms += time.perf_counter() - start

        best = ranked[0].text if ranked else ""
        snapped_counter.update(reference, best)
        snapped_counter.update_ranking(reference, [item.text for item in ranked])

        start = time.perf_counter()
        quick = ranker.suggest(roman, use_rules=True)
        quick_ms += time.perf_counter() - start
        rules_counter.update(reference, quick[0].text if quick else "")

        for cut in (3, 4, 5):
            prefix = roman[:cut]
            if len(prefix) < cut:
                continue
            completion_total[cut] += 1
            partial = engine.transliterate(prefix)
            words = lexicon.complete(partial, limit=8)
            if reference in words:
                completion_hits[cut] += 1

    total = max(len(rows), 1)
    result = {
        "file": Path(args.data).name,
        "n": len(rows),
        "model": model_counter.summary(),
        "snapped": snapped_counter.summary(),
        "rules_only": rules_counter.summary(),
        "completion": {
            str(cut): {
                "hit": completion_hits[cut],
                "n": completion_total[cut],
                "rate": round(completion_hits[cut] / max(completion_total[cut], 1), 4),
            }
            for cut in (3, 4, 5)
        },
        "latency_ms": {
            "model": round(model_ms * 1000 / total, 2),
            "snapped": round(snap_ms * 1000 / total, 2),
            "quick": round(quick_ms * 1000 / total, 2),
        },
    }

    print(f"model   top1 {model_counter.accuracy:.4f}  cer {model_counter.cer:.4f}")
    print(f"snapped top1 {snapped_counter.accuracy:.4f}  cer {snapped_counter.cer:.4f}")
    print(f"rules   top1 {rules_counter.accuracy:.4f}  cer {rules_counter.cer:.4f}")
    for cut, numbers in result["completion"].items():
        print(f"completion after {cut} letters: {numbers['rate']:.4f}")
    print(f"latency: {result['latency_ms']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

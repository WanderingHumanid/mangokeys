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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.lexicon import build_lexicon  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/demo/onnx-int8")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--data", default="data/processed/test_aksharantar.tsv")
    parser.add_argument("--train", default="data/processed/train.tsv")
    parser.add_argument("--frequency", default="data/raw/frequency/ml_50k.txt")
    parser.add_argument("--ngram", default="data/processed/ngram.txt")
    parser.add_argument("--lexicon-size", type=int, default=50000)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="rows to evaluate")
    parser.add_argument("--prefix-model", default="runs/prefix/onnx-int8")
    parser.add_argument("--weights", default="evals/results/weights.json")
    parser.add_argument("--out", default="evals/results/suggestions.json")
    args = parser.parse_args()

    rows = read_rows(Path(args.data), args.limit)
    print(f"rows: {len(rows)}")

    # The word list is built from the corpus and the frequency list only.
    # Nothing from the set under test goes into it.
    lexicon = build_lexicon(
        args.train,
        args.frequency,
        limit=args.lexicon_size,
    )
    print(f"lexicon: {len(lexicon)} words")
    covered = sum(1 for row in rows if row["native"] in lexicon)
    print(f"the word list knows {covered} of the {len(rows)} words under test")

    ngram = CharNgram.load(args.ngram) if Path(args.ngram).exists() else None
    print(f"language model: {'loaded' if ngram else 'missing'}")

    weights = Weights()
    if Path(args.weights).exists():
        saved = json.loads(Path(args.weights).read_text(encoding="utf-8"))["weights"]
        for name, value in saved.items():
            if hasattr(weights, name):
                setattr(weights, name, value)
        print(f"weights: {saved}")

    engine = Transliterator(args.onnx, args.tokenizer, beam=args.beam)
    prefix_engine = None
    if args.prefix_model and Path(args.prefix_model).exists():
        prefix_engine = Transliterator(args.prefix_model, args.tokenizer, beam=1)
        print(f"prefix model: {args.prefix_model}")
    ranker = SuggestionEngine(lexicon=lexicon, ngram=ngram, weights=weights)
    without_ngram = SuggestionEngine(lexicon=lexicon, ngram=None, weights=weights)

    model_counter = ErrorCounter()
    snapped_counter = ErrorCounter()
    no_ngram_counter = ErrorCounter()
    rules_counter = ErrorCounter()

    model_ms = 0.0
    snap_ms = 0.0
    quick_ms = 0.0
    completion_hits = {4: 0, 6: 0}
    completion_total = {4: 0, 6: 0}

    for number, row in enumerate(rows):
        roman = row["roman"]
        reference = row["native"]
        if number and number % 1000 == 0:
            print(f"  {number} of {len(rows)}", flush=True)

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

        plain = without_ngram.suggest(roman, texts, scores)
        no_ngram_counter.update(reference, plain[0].text if plain else "")

        start = time.perf_counter()
        quick = ranker.suggest(roman, use_rules=True)
        quick_ms += time.perf_counter() - start
        rules_counter.update(reference, quick[0].text if quick else "")

        for cut in (4, 6):
            prefix = roman[:cut]
            if len(prefix) < cut:
                continue
            completion_total[cut] += 1
            if prefix_engine is not None:
                partial = prefix_engine.transliterate(prefix)
            else:
                partial = engine.transliterate(prefix)
            words = lexicon.complete(partial, limit=8)
            if reference in words:
                completion_hits[cut] += 1

    total = max(len(rows), 1)
    result = {
        "file": Path(args.data).name,
        "n": len(rows),
        "dictionary_coverage": round(covered / max(len(rows), 1), 4),
        "model": model_counter.summary(),
        "with_word_list": snapped_counter.summary(),
        "without_language_model": no_ngram_counter.summary(),
        "rules_only": rules_counter.summary(),
        "completion": {
            str(cut): {
                "hit": completion_hits[cut],
                "n": completion_total[cut],
                "rate": round(completion_hits[cut] / max(completion_total[cut], 1), 4),
            }
            for cut in (4, 6)
        },
        "latency_ms": {
            "model": round(model_ms * 1000 / total, 2),
            "with_word_list": round(snap_ms * 1000 / total, 2),
            "quick": round(quick_ms * 1000 / total, 2),
        },
        "weights": {
            name: getattr(weights, name)
            for name in ("model", "known", "completion", "variation", "language", "frequency")
        },
    }

    print(f"model only          top1 {model_counter.accuracy:.4f}  cer {model_counter.cer:.4f}")
    print(f"with the word list  top1 {snapped_counter.accuracy:.4f}  cer {snapped_counter.cer:.4f}")
    print(f"no language model   top1 {no_ngram_counter.accuracy:.4f}")
    print(f"rules only          top1 {rules_counter.accuracy:.4f}  cer {rules_counter.cer:.4f}")
    for cut, numbers in result["completion"].items():
        print(f"completion after {cut} letters: {numbers['rate']:.4f}")
    print(f"latency: {result['latency_ms']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

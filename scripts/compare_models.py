"""Compare the transliteration model with the model that was trained on prefixes.

    python -m scripts.compare_models

The prefix model is trained on the same words plus the pairs where the roman
side is cut short. That buys the completion, a keyboard that can finish a
word from four letters. It costs something on the whole words, and this
script measures both sides of that trade on the same test set.
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


def measure(
    name: str,
    onnx: str,
    tokenizer: str,
    rows: list[tuple[str, str]],
    lexicon,
    beam: int,
    cuts: tuple[int, ...],
) -> dict:
    engine = Transliterator(onnx, tokenizer, beam=beam)
    counter = ErrorCounter()
    hits = {cut: 0 for cut in cuts}
    start = time.time()
    for number, (roman, reference) in enumerate(rows):
        candidates = engine.candidates(roman, n_best=5)
        best = candidates[0] if candidates else ""
        counter.update(reference, best)
        counter.update_ranking(reference, candidates)
        if number and number % 2000 == 0:
            print(f"    {name}: {number} of {len(rows)}", flush=True)
    whole = time.time() - start

    start = time.time()
    for number, (roman, reference) in enumerate(rows):
        for cut in cuts:
            prefix = roman[:cut]
            if len(prefix) < cut:
                continue
            partial = engine.transliterate(prefix)
            if reference in lexicon.complete(partial, limit=8):
                hits[cut] += 1
        if number and number % 2000 == 0:
            print(f"    {name}: completion {number} of {len(rows)}", flush=True)
    partial_time = time.time() - start

    summary = counter.summary()
    summary["completion"] = {
        str(cut): round(hits[cut] / max(len(rows), 1), 4) for cut in cuts
    }
    summary["ms_per_word"] = round(whole * 1000 / max(len(rows), 1), 2)
    summary["completion_ms_per_word"] = round(partial_time * 1000 / max(len(rows), 1), 2)
    print(f"  {name}: {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", default="runs/demo/onnx-int8")
    parser.add_argument("--prefix", default="runs/prefix/onnx-int8")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--data", default="data/processed/test_aksharantar.tsv")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--out", default="evals/results/prefix_model.json")
    args = parser.parse_args()

    rows = read_rows(Path(args.data), args.limit)
    print(f"rows: {len(rows)}")
    lexicon = build_lexicon("data/processed/train.tsv", limit=0)
    print(f"lexicon: {len(lexicon)} words")

    results = {}
    for name, path in (("the transliteration model", args.plain), ("the prefix model", args.prefix)):
        if not Path(path).exists():
            print(f"skipping {name}, {path} is missing")
            continue
        results[name] = measure(name, path, args.tokenizer, rows, lexicon, args.beam, (4, 6))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

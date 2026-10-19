"""Compare the browser runtime against the python runtime.

    node runtime/js/test_runtime.mjs runs/ml_mini/onnx runs/ml_mini/tokenizer.json /tmp/js.json
    python -m scripts.check_runtimes --onnx runs/ml_mini/onnx --js /tmp/js.json

The two runtimes drive the same two graphs the same way, so they should return
the same candidates. If they do not, one of them has a bug and the demo and the
python tooling would disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.python.transliterator import Transliterator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/ml_mini/onnx")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--js", required=True, help="the json the node script wrote")
    args = parser.parse_args()

    engine = Transliterator(args.onnx, args.tokenizer, max_src=48, max_tgt=64, beam=4)
    expected = json.loads(Path(args.js).read_text(encoding="utf-8"))

    mismatches = 0
    reordered = 0
    for word, js_candidates in expected.items():
        py_candidates = engine.candidates(word, n_best=5)
        if py_candidates == js_candidates:
            print(f"{word:24} same  {py_candidates[0] if py_candidates else ''}")
            continue
        if py_candidates[:1] != js_candidates[:1]:
            mismatches += 1
            print(f"{word}")
            print(f"  python {py_candidates}")
            print(f"  node   {js_candidates}")
        elif sorted(py_candidates) == sorted(js_candidates):
            # The same candidates in a different order, which happens when two
            # of them score the same and the two sorters break the tie
            # differently.
            reordered += 1
            print(f"{word:24} same set, different order")
        else:
            mismatches += 1
            print(f"{word}")
            print(f"  python {py_candidates}")
            print(f"  node   {js_candidates}")

    total = len(expected)
    print(f"\nchecked {total} words, {mismatches} mismatches, {reordered} reordered")
    if mismatches:
        raise SystemExit(1)
    print("the two runtimes agree on the top candidate")


if __name__ == "__main__":
    main()

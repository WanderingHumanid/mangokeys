"""Command line demo for the transliteration model.

    python -m scripts.demo --onnx runs/ml_mini/onnx --tokenizer runs/ml_mini/tokenizer.json

Type a Manglish word and see the candidates the model produces, next to what
the rule based engine gives for the same input.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.rules import RuleEngine  # noqa: E402
from runtime.python.transliterator import Transliterator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/ml_mini/onnx")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--word", default=None, help="transliterate one word and exit")
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()

    engine = Transliterator(
        args.onnx,
        args.tokenizer,
        max_src=48,
        max_tgt=64,
        beam=args.beam,
    )
    rules = RuleEngine()

    def show(word: str) -> None:
        start = time.time()
        candidates = engine.candidates(word, n_best=5)
        elapsed = (time.time() - start) * 1000
        rule_candidates = rules.convert(word)[:3]
        print(f"  {word}")
        print(f"    model : {' | '.join(candidates)}")
        print(f"    rules : {' | '.join(rule_candidates)}")
        print(f"    {elapsed:.0f} ms")

    if args.word:
        show(args.word)
        return

    print("Manglish to Malayalam demo. Type a word, or a blank line to quit.")
    while True:
        try:
            word = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not word:
            break
        show(word)


if __name__ == "__main__":
    main()

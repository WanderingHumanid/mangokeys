"""The keyboard, with the model, the word list and the language model.

    from keyboard import Keyboard

    keyboard = Keyboard(
        "runs/demo/onnx-int8",
        "runs/ml_mini/tokenizer.json",
        "data/processed/lexicon",
        "data/processed/ngram.txt",
    )
    keyboard.suggest("keral")

This is the piece the browser and the phone use. It has two paths.

`quick` uses the rules and the word list only. It takes well under a
millisecond, so it can run on every keystroke.

`suggest` adds the model. It takes a few milliseconds, so it runs when the
typing pauses or when the word is committed.

Both paths end in the same place. The word list turns whatever was proposed
into real words, and the language model ranks them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.lexicon import Lexicon  # noqa: E402
from model.src.ngram import CharNgram  # noqa: E402
from model.src.suggest import SuggestionEngine, Weights  # noqa: E402
from runtime.python.transliterator import Transliterator  # noqa: E402


class Keyboard:
    """The whole system behind one call."""

    def __init__(
        self,
        onnx_dir: str | Path,
        tokenizer_path: str | Path,
        lexicon_dir: str | Path | None = None,
        ngram_path: str | Path | None = None,
        beam: int = 4,
        limit: int = 8,
        weights: Weights | None = None,
    ) -> None:
        self.transliterator = Transliterator(
            onnx_dir, tokenizer_path, max_src=48, max_tgt=64, beam=beam
        )
        self.lexicon = Lexicon.load(lexicon_dir) if lexicon_dir else None
        self.ngram = CharNgram.load(ngram_path) if ngram_path else None
        self.engine = SuggestionEngine(
            lexicon=self.lexicon or Lexicon([], []),
            ngram=self.ngram,
            weights=weights or Weights(limit=limit),
        )

    def transliterate(self, roman: str) -> str:
        """The Malayalam of a word, without the word list."""
        return self.transliterator.transliterate(roman)

    def quick(self, roman: str, n_best: int = 8) -> list[str]:
        """The suggestions from the rules and the word list alone.

        This is the path that runs on every keystroke. It never calls the
        model, so it stays in the same millisecond for the whole word.
        """
        if not self.lexicon:
            return []
        ranked = self.engine.suggest(roman, use_rules=True)
        return [item.text for item in ranked[:n_best]]

    def suggest(self, roman: str, n_best: int = 8) -> list[str]:
        """The suggestions with the model in the loop."""
        candidates = self.transliterator.scored_candidates(roman, n_best)
        texts = [text for text, _ in candidates]
        scores = [score for _, score in candidates]
        ranked = self.engine.suggest(roman, texts, scores)
        return [item.text for item in ranked[:n_best]]

    def scored_suggestions(self, roman: str, n_best: int = 8):
        """The suggestions with the score of every one of them."""
        candidates = self.transliterator.scored_candidates(roman, n_best)
        texts = [text for text, _ in candidates]
        scores = [score for _, score in candidates]
        return self.engine.suggest(roman, texts, scores)[:n_best]

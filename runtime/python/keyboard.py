"""The keyboard, with the model, the word list and the language model.

    from keyboard import Keyboard

    keyboard = Keyboard(
        "runs/demo/onnx-int8",
        "runs/ml_mini/tokenizer.json",
        "data/processed/lexicon",
        prefix_model="runs/prefix/onnx-int8",
    )
    keyboard.suggest("keral")

This is the piece the browser and the phone use. It has two paths.

`quick` uses the rules and the word list only. It takes well under a
millisecond, so it can run on every keystroke.

`suggest` adds the models. It takes a few milliseconds, so it runs when the
typing pauses.

There are two models. The transliteration model is the one that was trained
on whole words, and it is the better of the two at them. The prefix model was
trained on the same words plus the pairs where the roman side is cut short,
so it learns to stop in the middle of a word, and the word list completes it.
Neither model is good at the other's job, and both are 6 MB.
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
        prefix_model: str | Path | None = None,
        beam: int = 4,
        limit: int = 8,
        weights: Weights | None = None,
    ) -> None:
        self.transliterator = Transliterator(
            onnx_dir, tokenizer_path, max_src=48, max_tgt=64, beam=beam
        )
        self.prefix_model = None
        if prefix_model is not None:
            self.prefix_model = Transliterator(
                prefix_model, tokenizer_path, max_src=48, max_tgt=64, beam=1
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

    def completion(self, roman: str, n_best: int = 8) -> list[str]:
        """The words that start with what the prefix model said.

        This is the "type four letters and get the whole word" behaviour.
        """
        if not self.lexicon or not self.prefix_model:
            return []
        prefix = self.prefix_model.transliterate(roman)
        if not prefix:
            return []
        return self.lexicon.complete(prefix, limit=n_best)

    def suggest(self, roman: str, n_best: int = 8) -> list[str]:
        """The suggestions with the models in the loop."""
        candidates = self.transliterator.scored_candidates(roman, 5)
        texts = [text for text, _ in candidates]
        scores = [score for _, score in candidates]
        completion = self.completion(roman, n_best=8)
        ranked = self.engine.suggest(roman, texts, scores, completion)
        return [item.text for item in ranked[:n_best]]

    def scored_suggestions(self, roman: str, n_best: int = 8):
        """The suggestions with the score of every one of them."""
        candidates = self.transliterator.scored_candidates(roman, 5)
        texts = [text for text, _ in candidates]
        scores = [score for _, score in candidates]
        completion = self.completion(roman, n_best=8)
        return self.engine.suggest(roman, texts, scores, completion)[:n_best]

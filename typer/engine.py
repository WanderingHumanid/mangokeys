"""Transliteration engine interface for MangoKeys Typer.

Seamlessly bridges between:
1. Instant zero-dependency rule-based conversion (model.src.rules.RuleEngine)
2. High-accuracy neural seq2seq ONNX model (runtime.python.transliterator.Transliterator)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path so model and runtime can be imported
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typer.config import DEFAULT_CONFIG  # noqa: E402


@dataclass
class TransliterationResult:
    """Holds the result of a single word transliteration."""

    input_text: str
    top_candidate: str
    candidates: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    engine_name: str = "rules"


class DesktopTransliterator:
    """Hybrid transliteration engine for the desktop daemon.

    Defaults to `RuleEngine` for zero latency and zero external dependencies,
    and upgrades dynamically to the neural ONNX model when model weights are
    detected and dependencies (onnxruntime, numpy) are present.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        force_engine: str | None = None,
    ) -> None:
        self.engine_type: str = "rules"
        self.model_desc: str = "RuleEngine (model.src.rules)"
        self._ml_transliterator: Any = None
        self._rule_engine: Any = None

        # Always initialize RuleEngine as the rock-solid baseline
        self._init_rule_engine()

        # If not explicitly locked to rules, attempt ML initialization
        if force_engine != "rules":
            self._try_init_ml(model_dir=model_dir, tokenizer_path=tokenizer_path)

        if force_engine == "onnx" and self.engine_type != "onnx":
            raise RuntimeError(
                f"Requested engine 'onnx' could not be loaded. Please ensure ONNX files "
                f"exist and onnxruntime/numpy are installed. Description: {self.model_desc}"
            )

    def _init_rule_engine(self) -> None:
        """Instantiate the rule-based transliterator."""
        try:
            from model.src.rules import RuleEngine

            self._rule_engine = RuleEngine(max_candidates=10)
            self.engine_type = "rules"
            self.model_desc = "RuleEngine (Deterministic Baseline)"
        except Exception as err:
            raise RuntimeError(f"Failed to initialize RuleEngine baseline: {err}") from err

    def _try_init_ml(
        self,
        model_dir: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> bool:
        """Probe for ONNX model weights and tokenizer, then attempt loading."""
        candidate_model_dir: Path | None = None
        candidate_tok_path: Path | None = None

        if model_dir:
            p = Path(model_dir)
            if p.is_dir() and (p / "encoder.onnx").exists() and (p / "decoder.onnx").exists():
                candidate_model_dir = p
        else:
            for rel in DEFAULT_CONFIG.ONNX_SEARCH_DIRS:
                p = _REPO_ROOT / rel
                if p.is_dir() and (p / "encoder.onnx").exists() and (p / "decoder.onnx").exists():
                    candidate_model_dir = p
                    break

        if tokenizer_path:
            p = Path(tokenizer_path)
            if p.is_file():
                candidate_tok_path = p
        else:
            for rel in DEFAULT_CONFIG.TOKENIZER_SEARCH_PATHS:
                p = _REPO_ROOT / rel
                if p.is_file():
                    candidate_tok_path = p
                    break

        # If files don't exist, keep rule engine fallback
        if not candidate_model_dir or not candidate_tok_path:
            return False

        try:
            from runtime.python.transliterator import Transliterator

            self._ml_transliterator = Transliterator(
                onnx_dir=candidate_model_dir,
                tokenizer_path=candidate_tok_path,
            )
            self.engine_type = "onnx"
            self.model_desc = f"Neural ONNX ({candidate_model_dir.name})"
            return True
        except Exception:
            # Graceful fallback: ML engine failed to load or dependencies missing
            return False

    def transliterate(self, text: str, n_best: int = 5) -> TransliterationResult:
        """Transliterate a single Latin word to Malayalam with candidate list and timing."""
        text = text.strip()
        if not text:
            return TransliterationResult(
                input_text=text,
                top_candidate="",
                candidates=[],
                latency_ms=0.0,
                engine_name=self.engine_type,
            )

        start_time = time.perf_counter()

        top_candidate = ""
        candidates: list[str] = []

        if self.engine_type == "onnx" and self._ml_transliterator is not None:
            try:
                candidates = self._ml_transliterator.candidates(text, n_best=n_best)
                if candidates:
                    top_candidate = candidates[0]
            except Exception:
                # Fallback to rules if inference encounters an issue
                pass

        # If ONNX didn't produce candidates or engine is rules
        if not top_candidate and self._rule_engine is not None:
            rule_candidates = self._rule_engine.convert(text)
            candidates = rule_candidates[:n_best]
            top_candidate = candidates[0] if candidates else text

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TransliterationResult(
            input_text=text,
            top_candidate=top_candidate,
            candidates=candidates,
            latency_ms=round(latency_ms, 2),
            engine_name=self.engine_type,
        )

    def convert_top(self, text: str) -> str:
        """Convenience method returning only the top Malayalam string."""
        return self.transliterate(text, n_best=1).top_candidate

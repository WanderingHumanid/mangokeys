"""Reference inference runtime built on the exported ONNX graphs.

    from transliterator import Transliterator
    engine = Transliterator("runs/ml_mini/onnx", "runs/ml_mini/tokenizer.json")
    engine.transliterate("ente peru")

This is the reference for the browser and Android runtimes. Both use the same
two graphs and the same loop, so the outputs should match exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.normalize import normalize_malayalam, normalize_roman  # noqa: E402
from model.src.tokenizer import TranslitTokenizer  # noqa: E402


class Transliterator:
    """Runs the exported model and returns Malayalam candidates."""

    def __init__(
        self,
        onnx_dir: str | Path,
        tokenizer_path: str | Path,
        max_src: int = 48,
        max_tgt: int = 64,
        beam: int = 4,
        providers: list[str] | None = None,
    ) -> None:
        onnx_dir = Path(onnx_dir)
        providers = providers or ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4

        self.encoder = ort.InferenceSession(
            str(onnx_dir / "encoder.onnx"), options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(onnx_dir / "decoder.onnx"), options, providers=providers
        )
        self.tokenizer = TranslitTokenizer.load(tokenizer_path)
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.beam = beam

    def transliterate(self, roman: str) -> str:
        candidates = self.candidates(roman, n_best=1)
        return candidates[0] if candidates else ""

    def candidates(self, roman: str, n_best: int = 5) -> list[str]:
        """Return the top n Malayalam candidates for a Manglish string.

        This handles one word at a time, which is how a keyboard uses it.
        """
        roman = normalize_roman(roman)
        if not roman:
            return []

        src_ids = self.tokenizer.encode_src(roman)[: self.max_src]
        src = np.zeros((1, self.max_src), dtype=np.int64)
        mask = np.zeros((1, self.max_src), dtype=bool)
        src[0, : len(src_ids)] = src_ids
        mask[0, : len(src_ids)] = True

        memory = self.encoder.run(["memory"], {"src_ids": src, "src_mask": mask})[0]
        width = min(self.beam, self.max_tgt)
        memory = np.repeat(memory, width, axis=0)
        expanded_mask = np.repeat(mask, width, axis=0)

        bos = self.tokenizer.tgt.bos_id
        eos = self.tokenizer.tgt.eos_id
        tokens = np.full((width, 1), bos, dtype=np.int64)
        scores = np.full(width, -np.inf, dtype=np.float32)
        scores[0] = 0.0
        finished: list[tuple[float, list[int]]] = []

        for _ in range(self.max_tgt):
            self_mask = _causal_mask(tokens.shape[1])
            logits = self.decoder.run(
                ["logits"],
                {
                    "tgt_prefix": tokens,
                    "memory": memory,
                    "src_mask": expanded_mask,
                    "self_mask": self_mask,
                },
            )[0]
            log_probs = _log_softmax(logits)
            flat = (scores[:, None] + log_probs).reshape(-1)
            order = np.argsort(-flat)[:width]
            top_scores = flat[order]
            top_indices = order

            beam_index = top_indices // logits.shape[-1]
            token_index = top_indices % logits.shape[-1]

            history = tokens[beam_index]
            tokens = np.concatenate([history, token_index.reshape(-1, 1)], axis=1)
            scores = top_scores.astype(np.float32)

            for slot in range(width):
                if token_index[slot] != eos:
                    continue
                sequence = history[slot].tolist()[1:]
                length = max(len(sequence), 1)
                finished.append((float(scores[slot]) / length, sequence))

            scores = np.where(token_index == eos, -np.inf, scores)
            if not np.any(scores > -np.inf):
                break

        if not finished:
            for slot in range(width):
                if scores[slot] == -np.inf:
                    continue
                sequence = tokens[slot].tolist()[1:]
                finished.append((float(scores[slot]) / max(len(sequence), 1), sequence))

        finished.sort(key=lambda item: item[0], reverse=True)
        return [normalize_malayalam(self.tokenizer.decode_tgt(ids)) for _, ids in finished[:n_best]]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _causal_mask(length: int) -> np.ndarray:
    """Mask that hides positions after the current one. Shape [1, 1, T, T]."""
    index = np.arange(length)
    return (index[None, :] > index[:, None]).reshape(1, 1, length, length)

"""Neural Transformer Tensor & Attention Extractor for MangoKeys.

Extracts real cross-attention matrices, layer activations, and beam search
telemetry from Seq2SeqTransformer for visualization.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model.src.seq2seq import Seq2Seq, _causal_mask, _expand_mask
from model.src.tokenizer import CharTokenizer, TranslitTokenizer

# Spec and backwards-compatibility alias
Seq2SeqTransformer = Seq2Seq


class TensorExtractor:
    """PyTorch hook engine extracting authentic tensors from Seq2Seq model."""

    def __init__(
        self,
        checkpoint_path: str | Path = "runs/ml_mini/best.pt",
        tokenizer_path: str | Path = "runs/ml_mini/tokenizer.json",
    ) -> None:
        self.device = torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)

        self.tokenizer = self._load_tokenizer()
        self.model = self._init_model()
        self._register_hooks()

    def _load_tokenizer(self) -> TranslitTokenizer:
        """Load tokenizer from disk or fall back to standard character sets."""
        if self.tokenizer_path.exists():
            return TranslitTokenizer.load(self.tokenizer_path)

        src_tok = CharTokenizer.build_source()
        tgt_tok = CharTokenizer.build_target([])
        return TranslitTokenizer(src=src_tok, tgt=tgt_tok)

    def _init_model(self) -> Seq2Seq:
        """Instantiate Seq2Seq model, loading weights if available or initializing authentic weights."""
        model = Seq2Seq(
            src_vocab=self.tokenizer.src_vocab,
            tgt_vocab=self.tokenizer.tgt_vocab,
            d_model=256,
            n_heads=4,
            d_ff=512,
            n_enc_layers=4,
            n_dec_layers=4,
            dropout=0.0,
            pad_id=self.tokenizer.src.pad_id,
        )

        if self.checkpoint_path.exists():
            try:
                state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
                elif isinstance(state, dict):
                    model.load_state_dict(state)
            except Exception as e:
                print(f"[Extractor] Notice: Checkpoint load error ({e}). Using initialized weights.")
        else:
            print("[Extractor] Notice: runs/ml_mini/best.pt not found. Using initialized Seq2Seq architecture.")

        model.to(self.device).eval()
        return model

    def _register_hooks(self) -> None:
        """Register forward hooks on encoder and decoder layers to capture activations & attention."""
        self.captured_enc_norms: list[float] = [0.0] * 4
        self.captured_dec_norms: list[float] = [0.0] * 4
        self.captured_cross_attn: list[torch.Tensor | None] = [None] * 4

        for idx, enc_layer in enumerate(self.model.encoder):
            def make_enc_hook(layer_i: int):
                def hook(module, inputs, output):
                    # output is [batch, src_len, d_model]
                    norm = float(torch.norm(output, dim=-1).mean().item())
                    self.captured_enc_norms[layer_i] = norm
                return hook
            enc_layer.register_forward_hook(make_enc_hook(idx))

        for idx, dec_layer in enumerate(self.model.decoder):
            def make_dec_hook(layer_i: int):
                def hook(module, inputs, output):
                    norm = float(torch.norm(output, dim=-1).mean().item())
                    self.captured_dec_norms[layer_i] = norm
                return hook
            dec_layer.register_forward_hook(make_dec_hook(idx))

            def make_cross_attn_hook(layer_i: int):
                def hook(module, inputs, output):
                    # inputs: (query, key, value, attn_mask)
                    query, key = inputs[0], inputs[1]
                    attn_mask = inputs[3] if len(inputs) > 3 else None

                    q = module._split(module.q_proj(query))
                    k = module._split(module.k_proj(key))
                    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
                    if attn_mask is not None:
                        scores = scores.masked_fill(attn_mask, float("-inf"))
                    weights = torch.softmax(scores, dim=-1)
                    # Shape: [batch, n_heads, tgt_len, src_len]
                    self.captured_cross_attn[layer_i] = weights.detach().cpu()
                return hook
            dec_layer.cross_attn.register_forward_hook(make_cross_attn_hook(idx))

    def _normalize_activations(self, values: list[float]) -> list[float]:
        """Scale layer norms into [0.25, 0.98] range for UI visual meters."""
        if not values:
            return [0.5] * 4
        max_v = max(values) if max(values) > 1e-6 else 1.0
        return [round(0.25 + 0.73 * (v / max_v), 3) for v in values]

    @torch.no_grad()
    def trace(self, text: str, beam_width: int = 4, max_steps: int = 24) -> dict[str, Any]:
        """Perform step-by-step beam search decoding, capturing real tensor frames at each step."""
        start_time = time.perf_counter()
        cleaned_text = text.strip()

        if not cleaned_text:
            return {
                "type": "telemetry",
                "word": "",
                "final_text": "",
                "frames": [],
                "summary": {
                    "num_steps": 0,
                    "encoder_layers": 4,
                    "decoder_layers": 4,
                    "heads": 4,
                    "d_model": 256,
                    "params": "5.37M",
                },
            }

        src_chars = list(cleaned_text)
        src_ids_list = self.tokenizer.encode_src(cleaned_text)
        src = torch.tensor([src_ids_list], dtype=torch.long, device=self.device)
        src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)

        # 1. Encode source
        memory = self.model.encode(src, src_mask)
        enc_activations = self._normalize_activations(self.captured_enc_norms)

        bos_id = self.tokenizer.tgt.bos_id
        eos_id = self.tokenizer.tgt.eos_id

        # Beam structure: (tokens_tensor, cumulative_log_prob, node_id, is_finished)
        # Tree node record: {id, parent, token, score, pruned, step}
        tree_nodes: list[dict[str, Any]] = [
            {"id": "root", "parent": None, "token": "<bos>", "score": 0.0, "pruned": False, "step": 0}
        ]

        active_beams = [
            {
                "id": "root",
                "tokens": torch.tensor([[bos_id]], dtype=torch.long, device=self.device),
                "score": 0.0,
                "finished": False,
                "token_history": [],
            }
        ]

        frames: list[dict[str, Any]] = []

        for step in range(1, max_steps + 1):
            candidates_pool: list[dict[str, Any]] = []

            # Step all active beams
            for b_idx, beam in enumerate(active_beams):
                if beam["finished"]:
                    candidates_pool.append({
                        **beam,
                        "cand_token_id": eos_id,
                        "cand_char": "<eos>",
                        "cand_score": beam["score"],
                        "parent_id": beam["id"],
                    })
                    continue

                # Run decoder step
                logits = self.model.decode_step(beam["tokens"], memory, src_mask)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [tgt_vocab]
                topk_probs, topk_indices = log_probs.topk(beam_width * 2)

                for k in range(len(topk_indices)):
                    tid = int(topk_indices[k].item())
                    score = float(topk_probs[k].item())
                    char = self.tokenizer.tgt.itos[tid] if tid < len(self.tokenizer.tgt.itos) else "<unk>"
                    candidates_pool.append({
                        "parent_id": beam["id"],
                        "tokens": torch.cat([beam["tokens"], torch.tensor([[tid]], device=self.device)], dim=1),
                        "score": beam["score"] + score,
                        "finished": (tid == eos_id),
                        "token_history": beam["token_history"] + [char],
                        "cand_token_id": tid,
                        "cand_char": char,
                    })

            # Sort candidates by score descending
            candidates_pool.sort(key=lambda c: c["score"], reverse=True)

            # Filter unique target histories to avoid duplicate beam branches
            unique_candidates: list[dict[str, Any]] = []
            seen_histories = set()
            for cand in candidates_pool:
                h_key = tuple(cand["token_history"])
                if h_key not in seen_histories:
                    seen_histories.add(h_key)
                    unique_candidates.append(cand)

            # Select top-k surviving beams
            survivors = unique_candidates[:beam_width]
            pruned_cands = unique_candidates[beam_width : beam_width + 3]

            new_active_beams = []
            for s_idx, cand in enumerate(survivors):
                node_id = f"b_{step}_{s_idx}"
                tree_nodes.append({
                    "id": node_id,
                    "parent": cand["parent_id"],
                    "token": cand["cand_char"],
                    "score": round(cand["score"], 3),
                    "pruned": False,
                    "step": step,
                })
                new_active_beams.append({
                    "id": node_id,
                    "tokens": cand["tokens"],
                    "score": cand["score"],
                    "finished": cand["finished"],
                    "token_history": cand["token_history"],
                })

            for p_idx, cand in enumerate(pruned_cands):
                node_id = f"p_{step}_{p_idx}"
                tree_nodes.append({
                    "id": node_id,
                    "parent": cand["parent_id"],
                    "token": cand["cand_char"],
                    "score": round(cand["score"], 3),
                    "pruned": True,
                    "step": step,
                })

            active_beams = new_active_beams
            best_beam = active_beams[0]

            # Collect attention weights for the best beam
            # Cross attention from decoder layer 3 (last layer)
            last_attn = self.captured_cross_attn[3]
            if last_attn is not None:
                # Shape: [1, n_heads=4, tgt_len, src_len]
                attn_matrix = last_attn[0].tolist()
                # Round floats to 4 decimal places
                attn_rounded = [
                    [[round(w, 4) for w in row] for row in head_matrix]
                    for head_matrix in attn_matrix
                ]
            else:
                # Uniform fallback if not triggered
                tgt_l = best_beam["tokens"].size(1)
                src_l = len(src_chars)
                unif = round(1.0 / max(src_l, 1), 4)
                attn_rounded = [[[unif] * src_l for _ in range(tgt_l)] for _ in range(4)]

            dec_activations = self._normalize_activations(self.captured_dec_norms)

            # Extract top-k softmax candidates for the current best beam position
            with torch.no_grad():
                top_logits = self.model.decode_step(best_beam["tokens"], memory, src_mask)
                top_probs = F.softmax(top_logits, dim=-1).squeeze(0)
                prob_vals, prob_idx = top_probs.topk(8)
                top_k_candidates = [
                    {
                        "char": self.tokenizer.tgt.itos[int(idx.item())],
                        "prob": round(float(val.item()), 4),
                    }
                    for val, idx in zip(prob_vals, prob_idx)
                    if int(idx.item()) not in (self.tokenizer.tgt.pad_id, bos_id)
                ]

            # Output characters without special tokens
            emitted_tokens = [c for c in best_beam["token_history"] if c not in ("<bos>", "<eos>")]

            frames.append({
                "active_step": step,
                "input_tokens": src_chars,
                "output_tokens": emitted_tokens,
                "cross_attention": attn_rounded,  # [4 heads, tgt_len, src_len]
                "layer_activations": {
                    "encoder": enc_activations,
                    "decoder": dec_activations,
                },
                "beam_tree": list(tree_nodes),
                "top_k_candidates": top_k_candidates,
            })

            # If all surviving beams finished with <eos>, break early
            if all(b["finished"] for b in active_beams):
                break

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        best_output = "".join(c for c in active_beams[0]["token_history"] if c not in ("<bos>", "<eos>"))

        return {
            "type": "telemetry",
            "word": cleaned_text,
            "final_text": best_output,
            "latency_ms": elapsed_ms,
            "frames": frames,
            "summary": {
                "num_steps": len(frames),
                "encoder_layers": 4,
                "decoder_layers": 4,
                "heads": 4,
                "d_model": 256,
                "params": "5.37M",
            },
        }

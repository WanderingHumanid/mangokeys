"""A compact character level Transformer encoder decoder for transliteration.

The encoder and decoder are kept as separate modules so that the exported ONNX
graph can run the encoder once and then step the decoder one token at a time.
That is what lets the host runtime do beam search without a Python loop inside
the graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions. Character sequences are short, so there is
    nothing to learn here."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(weights, v)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, d_ff)
        self.w_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.dropout(F.gelu(self.w_in(x))))


class EncoderLayer(nn.Module):
    """Pre-norm self attention block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Pre-norm block with masked self attention and cross attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), self_mask))
        x = x + self.dropout(
            self.cross_attn(self.norm2(x), self.norm2(memory), self.norm2(memory), cross_mask)
        )
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Seq2Seq(nn.Module):
    """Character level transliteration model.

    Source and target embeddings are separate because the two sides live in
    different scripts and share no useful statistics.
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        dropout: float = 0.3,
        max_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)

        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encode a padded source batch. Returns the memory tensor."""
        x = self.dropout(self.pos(self.src_embed(src_ids)))
        attn_mask = _expand_mask(src_mask)
        for layer in self.encoder:
            x = layer(x, attn_mask)
        return self.enc_norm(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a full target sequence in one shot. Used during training."""
        x = self.dropout(self.pos(self.tgt_embed(tgt_ids)))
        self_mask = _causal_mask(tgt_ids.size(1), tgt_ids.device)
        cross_mask = _expand_mask(src_mask)
        for layer in self.decoder:
            x = layer(x, memory, self_mask, cross_mask)
        return self.output_proj(self.dec_norm(x))

    def decode_step(
        self,
        tgt_last: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a single step given the last emitted token.

        The host runtime calls this in a loop, or in parallel across a beam.
        Returns logits of shape [batch, vocab].
        """
        x = self.dropout(self.pos(self.tgt_embed(tgt_last)))
        self_mask = None
        cross_mask = _expand_mask(src_mask)
        for layer in self.decoder:
            x = layer(x, memory, self_mask, cross_mask)
        return self.output_proj(self.dec_norm(x))[:, -1]

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, memory, src_mask)


def _expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """Turn a [B, S] key padding mask into a [B, 1, 1, S] additive mask."""
    return (mask == 0).view(mask.size(0), 1, 1, mask.size(1))


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Additive mask that hides future positions from every query."""
    future = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    return future.view(1, 1, length, length)


class LabelSmoothedLoss(nn.Module):
    """Cross entropy with label smoothing and padding ignored."""

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        target = target.reshape(-1)
        valid = target != self.pad_id
        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.sum() / valid.sum()

"""Datasets and batching for transliteration training.

Malayalam targets are noticeably longer than their roman source because every
matra is a separate codepoint. Batching by length saves a lot of wasted
compute, so the sampler groups examples of similar length together while still
shuffling the order between epochs.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from model.src.tokenizer import TranslitTokenizer


class PairDataset(Dataset):
    """Parallel roman and native text encoded to tensors.

    The encoded ids live in two flat numpy arrays with an offset table. A list
    of python lists would cost several gigabytes once the corpus reaches a few
    million pairs.
    """

    def __init__(
        self,
        pairs: list[dict],
        tokenizer: TranslitTokenizer,
        max_src: int = 48,
        max_tgt: int = 64,
    ) -> None:
        self.pairs = pairs
        self.max_src = max_src
        self.max_tgt = max_tgt

        src_chunks: list[np.ndarray] = []
        src_offsets = np.zeros(len(pairs) + 1, dtype=np.int64)
        tgt_chunks: list[np.ndarray] = []
        tgt_offsets = np.zeros(len(pairs) + 1, dtype=np.int64)

        for index, pair in enumerate(pairs):
            src = tokenizer.encode_src(pair["roman"])[:max_src]
            tgt = tokenizer.encode_tgt(pair["native"])[:max_tgt]
            src_chunks.append(np.asarray(src, dtype=np.int32))
            tgt_chunks.append(np.asarray(tgt, dtype=np.int32))
            src_offsets[index + 1] = src_offsets[index] + len(src)
            tgt_offsets[index + 1] = tgt_offsets[index] + len(tgt)

        self.src_flat = (
            np.concatenate(src_chunks) if src_chunks else np.zeros(0, dtype=np.int32)
        )
        self.tgt_flat = (
            np.concatenate(tgt_chunks) if tgt_chunks else np.zeros(0, dtype=np.int32)
        )
        self.src_offsets = src_offsets
        self.tgt_offsets = tgt_offsets

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        src = torch.from_numpy(self.src_flat[self.src_offsets[index] : self.src_offsets[index + 1]])
        tgt = torch.from_numpy(self.tgt_flat[self.tgt_offsets[index] : self.tgt_offsets[index + 1]])
        return src.long(), tgt.long()

    @property
    def lengths(self) -> list[int]:
        return np.diff(self.src_offsets).tolist()


class Collator:
    """Pads a batch and builds the teacher forcing inputs."""

    def __init__(self, tokenizer: TranslitTokenizer) -> None:
        self.pad_id = tokenizer.src.pad_id
        self.bos_id = tokenizer.tgt.bos_id
        self.eos_id = tokenizer.tgt.eos_id

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]

        src_len = max(len(ids) for ids in src_batch)
        tgt_len = max(len(ids) for ids in tgt_batch) + 1

        src = torch.full((len(batch), src_len), self.pad_id, dtype=torch.long)
        tgt_in = torch.full((len(batch), tgt_len), self.pad_id, dtype=torch.long)
        tgt_out = torch.full((len(batch), tgt_len), self.pad_id, dtype=torch.long)
        src_mask = torch.zeros((len(batch), src_len), dtype=torch.bool)

        for row, (src_ids, tgt_ids) in enumerate(zip(src_batch, tgt_batch, strict=True)):
            src[row, : len(src_ids)] = src_ids
            src_mask[row, : len(src_ids)] = True

            target = tgt_ids.tolist()
            tgt_in[row, 0] = self.bos_id
            tgt_in[row, 1 : len(target) + 1] = tgt_ids
            tgt_out[row, : len(target)] = tgt_ids
            tgt_out[row, len(target)] = self.eos_id

        return {"src": src, "src_mask": src_mask, "tgt_in": tgt_in, "tgt_out": tgt_out}


class BucketBatchSampler(Sampler[list[int]]):
    """Groups examples of similar length into batches.

    Examples are sorted inside large shuffled chunks, so batches are mostly
    homogeneous in length while the batch order still changes every epoch.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        chunk_mult: int = 50,
        seed: int = 0,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = batch_size * chunk_mult
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.chunk_size):
            chunk = indices[start : start + self.chunk_size]
            chunk.sort(key=lambda i: self.lengths[i])
            for offset in range(0, len(chunk), self.batch_size):
                batches.append(chunk[offset : offset + self.batch_size])

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

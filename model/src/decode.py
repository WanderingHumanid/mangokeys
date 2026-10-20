"""Decoding strategies for the transliteration model.

The exported ONNX graph exposes an encoder and a single decoder step, so the
runtimes in Python, JavaScript and Kotlin all drive decoding the same way:
encode once, then feed the last emitted token back in and collect the logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.src.seq2seq import Seq2Seq


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 64,
) -> list[list[int]]:
    """Decode a batch greedily. Returns one id list per source row."""
    model.eval()
    device = src_ids.device
    memory = model.encode(src_ids, src_mask)

    batch = src_ids.size(0)
    tokens = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(batch)]

    for _ in range(max_len):
        logits = model.decode_step(tokens, memory, src_mask)
        next_ids = logits.argmax(dim=-1)
        for row in range(batch):
            if finished[row]:
                continue
            token = int(next_ids[row])
            if token == eos_id:
                finished[row] = True
                continue
            outputs[row].append(token)
        if bool(finished.all()):
            break
        tokens = torch.cat([tokens, next_ids.unsqueeze(1)], dim=1)

    return outputs


@torch.no_grad()
def beam_search(
    model: Seq2Seq,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam: int = 4,
    max_len: int = 64,
    length_penalty: float = 0.0,
    n_best: int = 1,
) -> list[list[list[int]]]:
    """Beam search over a batch.

    Returns n_best candidate id lists per source row, ordered by score.

    The scores are raw log probabilities. A live beam can only add tokens, so
    its score can never go up, which makes it safe to drop a live beam that is
    already behind the best finished hypothesis and to stop the search when no
    live beam can still win. Without that the search keeps stepping beams long
    after the answer is known, and that is most of the latency.
    """
    model.eval()
    device = src_ids.device
    batch = src_ids.size(0)
    width = min(beam, max_len)

    memory = model.encode(src_ids, src_mask)
    memory = memory.repeat_interleave(width, dim=0)
    expanded_mask = src_mask.repeat_interleave(width, dim=0)

    tokens = torch.full((batch * width, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((batch, width), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)

    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch)]
    best_finished = torch.full((batch,), float("-inf"), device=device)

    for _ in range(max_len):
        logits = model.decode_step(tokens, memory, expanded_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        vocab = log_probs.size(-1)

        candidate = (scores.unsqueeze(1) + log_probs).view(batch, width * vocab)
        top_scores, top_indices = candidate.topk(width, dim=-1)

        # The parent of each candidate is a beam inside the same batch row, so
        # the row offset has to be added to get the flat index.
        row_offset = (torch.arange(batch, device=device) * width).view(batch, 1)
        beam_index = (top_indices // vocab + row_offset).view(-1)
        token_index = (top_indices % vocab).view(-1)

        history = tokens[beam_index]
        next_token = token_index.view(-1, 1)
        tokens = torch.cat([history, next_token], dim=1)
        scores = top_scores.view(-1)

        for flat in range(batch * width):
            if int(token_index[flat]) != eos_id:
                continue
            row = flat // width
            sequence = history[flat].tolist()[1:]
            score = float(scores[flat])
            finished[row].append((score, sequence))
            if score > float(best_finished[row]):
                best_finished[row] = score

        # A live beam can only lose score, so it is hopeless once it is behind
        # the best finished hypothesis.
        scores = scores.masked_fill(
            scores < best_finished.repeat_interleave(width), float("-inf")
        )
        if not bool((scores > float("-inf")).any()):
            break

    results: list[list[list[int]]] = []
    for row in range(batch):
        candidates = list(finished[row])
        if not candidates:
            # Nothing finished inside the length budget, fall back to the live beams.
            for slot in range(width):
                flat = row * width + slot
                score = float(scores[flat])
                if score == float("-inf"):
                    continue
                sequence = tokens[flat].tolist()[1:]
                length = max(len(sequence), 1)
                candidates.append((score / length**length_penalty, sequence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        results.append([sequence for _, sequence in candidates[:n_best]])

    return results

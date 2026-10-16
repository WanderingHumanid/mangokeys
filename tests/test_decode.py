"""Tests for greedy and beam decoding."""

import torch

from model.src.decode import beam_search, greedy_decode
from model.src.seq2seq import Seq2Seq


def build_model() -> Seq2Seq:
    torch.manual_seed(0)
    return Seq2Seq(
        src_vocab=99,
        tgt_vocab=132,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_enc_layers=1,
        n_dec_layers=1,
        dropout=0.0,
        pad_id=0,
    ).eval()


def test_greedy_returns_one_list_per_row():
    model = build_model()
    src = torch.randint(3, 90, (3, 6))
    mask = torch.ones(3, 6, dtype=torch.bool)
    out = greedy_decode(model, src, mask, bos_id=2, eos_id=3, max_len=10)
    assert len(out) == 3
    assert all(isinstance(row, list) for row in out)


def test_greedy_stops_at_eos():
    model = build_model()
    src = torch.randint(3, 90, (1, 6))
    mask = torch.ones(1, 6, dtype=torch.bool)
    out = greedy_decode(model, src, mask, bos_id=2, eos_id=3, max_len=10)
    assert 3 not in out[0]


def test_beam_returns_candidates():
    model = build_model()
    src = torch.randint(3, 90, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = beam_search(model, src, mask, bos_id=2, eos_id=3, beam=3, max_len=8, n_best=2)
    assert len(out) == 2
    assert all(len(row) <= 2 for row in out)


def test_beam_scores_are_ordered():
    model = build_model()
    src = torch.randint(3, 90, (1, 6))
    mask = torch.ones(1, 6, dtype=torch.bool)
    out = beam_search(model, src, mask, bos_id=2, eos_id=3, beam=4, max_len=8, n_best=4)
    assert len(out[0]) == 4
    assert len(set(len(c) for c in out[0])) == 1

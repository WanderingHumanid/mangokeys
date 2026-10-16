"""Tests for the seq2seq model."""

import torch

from model.src.seq2seq import Seq2Seq


def build(**kwargs) -> Seq2Seq:
    defaults = dict(
        src_vocab=99,
        tgt_vocab=132,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_enc_layers=2,
        n_dec_layers=2,
        dropout=0.0,
        pad_id=0,
    )
    defaults.update(kwargs)
    return Seq2Seq(**defaults)


def test_forward_shapes():
    model = build()
    src = torch.randint(3, 90, (2, 7))
    tgt = torch.randint(3, 120, (2, 5))
    mask = torch.ones(2, 7, dtype=torch.bool)
    out = model(src, tgt, mask)
    assert out.shape == (2, 5, 132)


def test_decoder_is_causal():
    model = build().eval()
    src = torch.randint(3, 90, (1, 6))
    mask = torch.ones(1, 6, dtype=torch.bool)
    tgt = torch.randint(3, 120, (1, 5))

    base = model(src, tgt, mask)
    changed = tgt.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 120
    other = model(src, changed, mask)

    assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], other[:, -1], atol=1e-5)


def test_padding_is_ignored():
    model = build().eval()
    src = torch.randint(3, 90, (1, 6))
    mask = torch.tensor([[True, True, True, False, False, False]])
    tgt = torch.randint(3, 120, (1, 4))

    base = model(src, tgt, mask)
    padded = src.clone()
    padded[0, 3:] = 55
    other = model(padded, tgt, mask)

    assert torch.allclose(base, other, atol=1e-5)


def test_decode_step_matches_full_decode():
    model = build().eval()
    src = torch.randint(3, 90, (1, 6))
    mask = torch.ones(1, 6, dtype=torch.bool)
    tgt = torch.randint(3, 120, (1, 4))

    memory = model.encode(src, mask)
    full = model.decode(tgt, memory, mask)
    stepped = model.decode_step(tgt[:, :1], memory, mask)

    assert torch.allclose(full[:, 0], stepped[:, 0], atol=1e-5)


def test_parameter_count_is_small():
    model = build(
        src_vocab=99,
        tgt_vocab=132,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_enc_layers=4,
        n_dec_layers=4,
    )
    params = sum(p.numel() for p in model.parameters())
    # Roughly 5.3M for the real vocabularies, well under 10M here.
    assert 4_500_000 < params < 6_000_000

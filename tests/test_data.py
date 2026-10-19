"""Tests for the dataset and batching."""

from model.src.data import BucketBatchSampler, Collator, PairDataset
from model.src.tokenizer import CharTokenizer, TranslitTokenizer


def build_tokenizer() -> TranslitTokenizer:
    src = CharTokenizer.build_source()
    tgt = CharTokenizer.build_target(["കേരളം", "സുഖമാണോ", "നിങ്ങൾക്കും"])
    return TranslitTokenizer(src=src, tgt=tgt)


def test_collator_shapes_and_teacher_forcing():
    tokenizer = build_tokenizer()
    pairs = [{"roman": "kerala", "native": "കേരളം"}, {"roman": "su", "native": "സു"}]
    dataset = PairDataset(pairs, tokenizer)
    collator = Collator(tokenizer)

    batch = collator([dataset[0], dataset[1]])
    assert batch["src"].shape == batch["src_mask"].shape
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["tgt_in"].shape[0] == 2


def test_collator_pads_and_masks():
    tokenizer = build_tokenizer()
    pairs = [{"roman": "kerala", "native": "കേരളം"}, {"roman": "su", "native": "സു"}]
    dataset = PairDataset(pairs, tokenizer)
    collator = Collator(tokenizer)

    batch = collator([dataset[0], dataset[1]])
    assert batch["src_mask"][1, 2:].sum() == 0
    assert batch["src"][1, 2:].unique().tolist() == [tokenizer.src.pad_id]


def test_targets_end_with_eos():
    tokenizer = build_tokenizer()
    dataset = PairDataset([{"roman": "su", "native": "സു"}], tokenizer)
    collator = Collator(tokenizer)
    batch = collator([dataset[0]])
    length = len(tokenizer.encode_tgt("സു")) + 1
    assert batch["tgt_out"][0, length - 1].item() == tokenizer.tgt.eos_id
    assert batch["tgt_in"][0, 0].item() == tokenizer.tgt.bos_id


def test_bucket_sampler_covers_everything_once():
    lengths = [3, 1, 2, 5, 4, 9, 7, 8, 6, 2]
    sampler = BucketBatchSampler(lengths, batch_size=3, shuffle=False)
    seen: list[int] = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))


def test_bucket_sampler_groups_similar_lengths():
    lengths = list(range(100))
    sampler = BucketBatchSampler(lengths, batch_size=10, shuffle=False, chunk_mult=1)
    for batch in sampler:
        spread = max(lengths[i] for i in batch) - min(lengths[i] for i in batch)
        assert spread < 10


def test_bucket_sampler_changes_order_between_epochs():
    lengths = list(range(100))
    sampler = BucketBatchSampler(lengths, batch_size=10, shuffle=True, seed=7)
    first = [tuple(batch) for batch in sampler]
    sampler.set_epoch(1)
    second = [tuple(batch) for batch in sampler]
    assert first != second

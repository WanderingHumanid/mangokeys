"""Character level tokenization for Manglish to Malayalam transliteration.

Malayalam is a matras heavy abugida, so a single grapheme can span two to four
codepoints. Subword tokenizers split those apart, which is exactly where
transliteration models tend to break. We work on characters and keep the
vocabulary closed and small.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)

# The whole Malayalam block, seeded so that no target character is ever unknown.
MALAYALAM_BLOCK = range(0x0D00, 0x0D80)

# Latin characters accepted on the source side.
UPPER = range(0x41, 0x5B)
LOWER = range(0x61, 0x7B)
DIGITS = range(0x30, 0x3A)
PUNCTUATION = set(" .,-_'!?/():;@&+=\"#*[]{}<>|~`\\^$%")


@dataclass
class CharTokenizer:
    """Maps characters to integer ids and back for one side of the pair."""

    itos: list[str]
    stoi: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos) and self.itos[i] not in SPECIALS)

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> CharTokenizer:
        """Build a tokenizer from a list of strings."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        vocab = list(SPECIALS) + sorted(
            (ch for ch, n in counter.items() if n >= min_freq),
            key=lambda ch: (-counter[ch], ch),
        )
        return cls(itos=vocab, stoi={ch: i for i, ch in enumerate(vocab)})

    @classmethod
    def build_source(cls) -> CharTokenizer:
        """Build the fixed source vocabulary.

        The source side is closed over printable ASCII so that anything a user
        can type is representable and never out of vocabulary.
        """
        chars = [chr(cp) for cp in UPPER]
        chars += [chr(cp) for cp in LOWER]
        chars += [chr(cp) for cp in DIGITS]
        chars += sorted(PUNCTUATION)
        itos = list(SPECIALS) + chars
        return cls(itos=itos, stoi={ch: i for i, ch in enumerate(itos)})

    @classmethod
    def build_target(cls, texts: list[str], min_freq: int = 1) -> CharTokenizer:
        """Build the target vocabulary, seeded with the Malayalam block."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)

        seed = [chr(cp) for cp in MALAYALAM_BLOCK]
        seed_set = set(seed)
        extra = sorted(
            (ch for ch, n in counter.items() if n >= min_freq and ch not in seed_set),
            key=lambda ch: (-counter[ch], ch),
        )
        itos = list(SPECIALS) + seed + extra
        return cls(itos=itos, stoi={ch: i for i, ch in enumerate(itos)})


@dataclass
class TranslitTokenizer:
    """The pair of tokenizers used by the model."""

    src: CharTokenizer
    tgt: CharTokenizer

    def encode_src(self, text: str) -> list[int]:
        return self.src.encode(text)

    def encode_tgt(self, text: str) -> list[int]:
        return self.tgt.encode(text)

    def decode_tgt(self, ids: list[int]) -> str:
        return self.tgt.decode(ids)

    @property
    def src_vocab(self) -> int:
        return len(self.src)

    @property
    def tgt_vocab(self) -> int:
        return len(self.tgt)

    def save(self, path: str | Path) -> None:
        payload = {"src": self.src.itos, "tgt": self.tgt.itos}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TranslitTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        src = CharTokenizer(itos=payload["src"], stoi={c: i for i, c in enumerate(payload["src"])})
        tgt = CharTokenizer(itos=payload["tgt"], stoi={c: i for i, c in enumerate(payload["tgt"])})
        return cls(src=src, tgt=tgt)

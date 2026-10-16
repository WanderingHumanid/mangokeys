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

# Latin ranges we accept on the source side.
LATIN_RANGES = (range(0x41, 0x5B), range(0x61, 0x7B), range(0x30, 0x3A))
PUNCTUATION = set(" .,-_'!?/():;@&+=\"#*[]{}<>|~`\\^$%")


@dataclass
class CharTokenizer:
    """Maps characters to integer ids and back."""

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

    def decode(self, ids: list[int], skip_specials: bool = True) -> str:
        out = []
        for i in ids:
            token = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if skip_specials and token in SPECIALS:
                continue
            out.append(token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        return cls(itos=itos, stoi={ch: i for i, ch in enumerate(itos)})

    @classmethod
    def build(
        cls,
        source_texts: list[str],
        target_texts: list[str],
        min_freq: int = 1,
    ) -> CharTokenizer:
        """Build a joint tokenizer from parallel text.

        The source side is closed over known Latin characters so that typing is
        never out of vocabulary. The target side is seeded with the full
        Malayalam block and then extended with anything else seen in the data.
        """
        source_vocab = _source_vocab()
        target_counter: Counter[str] = Counter()
        for text in target_texts:
            target_counter.update(text)

        target_vocab = [chr(cp) for cp in MALAYALAM_BLOCK]
        target_vocab += sorted(
            (ch for ch, n in target_counter.items() if n >= min_freq and ch not in target_vocab),
            key=lambda ch: (-target_counter[ch], ch),
        )

        itos = list(SPECIALS) + source_vocab + target_vocab
        return cls(itos=itos, stoi={ch: i for i, ch in enumerate(itos)})

    def split_vocab(self) -> tuple[list[str], list[str]]:
        """Return the source and target vocabularies as separate lists.

        The encoder and decoder do not share embeddings, so training needs to
        know where one vocabulary ends and the other begins.
        """
        source = [t for t in self.itos if t not in SPECIALS and _is_source_char(t)]
        target = [t for t in self.itos if t not in SPECIALS and not _is_source_char(t)]
        return source, target


def _is_source_char(ch: str) -> bool:
    if ch in LATIN_RANGES[0] or ch in LATIN_RANGES[1] or ch in LATIN_RANGES[2]:
        return True
    return ch in PUNCTUATION


def _source_vocab() -> list[str]:
    chars: list[str] = []
    for rng in LATIN_RANGES:
        chars.extend(chr(cp) for cp in rng)
    chars.extend(sorted(PUNCTUATION))
    return chars

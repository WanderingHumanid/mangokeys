"""Helpers for reading and writing transliteration pair files.

Pairs are stored as tab separated text so that they stay easy to inspect and
to diff. The roman form comes first, the native form second. An optional third
column carries a bucket label such as the test subset a row came from.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

ROMAN, NATIVE = "roman", "native"
BUCKET = "bucket"


def write_pairs(path: str | Path, pairs: Iterable[dict]) -> int:
    """Write pairs as TSV. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            label = pair.get(BUCKET, "")
            handle.write(f"{pair[ROMAN]}\t{pair[NATIVE]}\t{label}\n")
            count += 1
    return count


def read_pairs(path: str | Path) -> Iterator[dict]:
    """Read pairs from a TSV file, skipping malformed rows."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pair = {ROMAN: parts[0], NATIVE: parts[1]}
            if len(parts) > 2 and parts[2]:
                pair[BUCKET] = parts[2]
            yield pair


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

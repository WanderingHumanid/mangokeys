"""Download the Malayalam slice of the Aksharantar dataset.

Aksharantar ships one zip per language. The Malayalam one is about 125 MB and
holds roughly 4.1 million roman to native word pairs.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main"
# Aksharantar names its files with the ISO 639-2 code, so Malayalam is mal.
LANG = "mal"


def download_zip(out_dir: Path, lang: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{lang}.zip"
    if not archive.exists():
        url = f"{BASE}/{lang}.zip"
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    return archive


def extract(archive: Path, out_dir: Path) -> Path:
    target = out_dir / "aksharantar_ml"
    if not target.exists():
        print(f"extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    return target


def find_jsonl(root: Path) -> Path:
    # The archive stores json lines in files with a .json extension.
    candidates = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no json files under {root}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="destination directory")
    parser.add_argument("--lang", default=LANG, help="language code")
    args = parser.parse_args()

    out_dir = Path(args.out)
    archive = download_zip(out_dir, args.lang)
    root = extract(archive, out_dir)
    jsonl = find_jsonl(root)
    print(f"data file: {jsonl}")

    with jsonl.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    print(f"columns: {list(first.keys())}")
    print(f"example: {first}")


if __name__ == "__main__":
    main()

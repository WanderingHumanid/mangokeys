"""Download the Malayalam word frequency list.

    python -m data.scripts.download_frequency --out data/raw/frequency

The list is the one Hermit Dave publishes from the OpenSubtitles corpus. It
is the only source in this project that says how often a word is really
used, which is what the word list of the keyboard needs.

The licence is MIT, so the file can be used and redistributed.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ml/ml_50k.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/frequency")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "ml_50k.txt"
    if path.exists() and path.stat().st_size > 0:
        print(f"{path} is already there")
        return

    print(f"downloading {args.url}")
    urllib.request.urlretrieve(args.url, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"wrote {path} with {len(lines)} words")


if __name__ == "__main__":
    main()

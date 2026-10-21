"""Download the Malayalam lexicon of the Dakshina dataset.

    python -m data.scripts.download_dakshina_lexicon --out data/raw/dakshina

The lexicon is a list of Malayalam words with the number of times each one
was seen. It is the only source in this project with a real count of how
often a Malayalam word is used, apart from the frequency list of the
subtitles.

The licence of the Dakshina dataset is CC BY-SA 4.0, so the file can be used
and redistributed as long as the dataset is credited.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/vrclc/dakshina-lexicons-ml/resolve/main"
FILES = ("train.tsv", "valid.tsv", "test.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/dakshina")
    args = parser.parse_args()

    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = target / name
        if path.exists() and path.stat().st_size > 0:
            print(f"{path} is already there")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

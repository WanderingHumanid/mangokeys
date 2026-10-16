"""Download the Dakshina dataset and keep only the Malayalam subtree.

The full release is about 9 GB, but we only need the ml directory. The tar is
streamed so nothing large is ever written to disk.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
LANG = "ml"


def download(out_dir: Path, lang: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{lang}/"

    with urllib.request.urlopen(URL) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                if not member.name.startswith(prefix):
                    continue
                tar.extract(member, path=out_dir, filter="data")
                print(f"  extracted {member.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/dakshina", help="destination directory")
    parser.add_argument("--lang", default=LANG, help="language subdirectory to keep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    target = out_dir / args.lang
    if target.exists() and any(target.iterdir()):
        print(f"{target} already exists, skipping download")
        return

    print(f"streaming {URL} and extracting {args.lang}/ only")
    download(out_dir, args.lang)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()

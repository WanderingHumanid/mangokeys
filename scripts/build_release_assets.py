"""Assemble the model bundle that the release builds against.

    python -m scripts.build_release_assets

The Android app and the web page both need the same files: the transliteration
model, the prefix model, the tokenizer and the word list. A build on a clean
checkout has to find them somewhere, so they are collected into one directory
with a manifest of their checksums.

The files are copies, so this script never trains anything. It only gathers
what the other scripts already produced and describes what it gathered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The layout the build expects. The name of a folder is the name of the
# feature it holds, and the file names inside it are the names the runtime
# looks for.
LAYOUT = {
    "model": ("encoder.onnx", "decoder.onnx", "tokenizer.json"),
    "prefix": ("encoder.onnx", "decoder.onnx"),
    "lexicon": ("words.txt", "index.bin"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="assets/release")
    parser.add_argument("--model", default="runs/demo/onnx-int8")
    parser.add_argument("--tokenizer", default="runs/ml_mini/tokenizer.json")
    parser.add_argument("--prefix-model", default="runs/prefix/onnx-int8")
    parser.add_argument("--lexicon", default="data/processed/lexicon")
    args = parser.parse_args()

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)

    sources = {
        "model": {
            "encoder.onnx": ROOT / args.model / "encoder.onnx",
            "decoder.onnx": ROOT / args.model / "decoder.onnx",
            "tokenizer.json": ROOT / args.tokenizer,
        },
        "prefix": {
            "encoder.onnx": ROOT / args.prefix_model / "encoder.onnx",
            "decoder.onnx": ROOT / args.prefix_model / "decoder.onnx",
        },
        "lexicon": {
            "words.txt": ROOT / args.lexicon / "words.txt",
            "index.bin": ROOT / args.lexicon / "index.bin",
        },
    }

    missing = [
        f"{group}/{name}"
        for group, files in sources.items()
        for name, path in files.items()
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "these files are missing, build them first:\n  "
            + "\n  ".join(missing)
            + "\n\nsee docs/RELEASING.md for the order of the commands"
        )

    manifest = {"files": {}}
    for group, files in sources.items():
        for name, path in files.items():
            target = out / group / name
            copy(path, target)
            manifest["files"][f"{group}/{name}"] = {
                "bytes": target.stat().st_size,
                "sha256": digest(target),
            }
            print(f"{group}/{name}: {target.stat().st_size / 1e6:.2f} MB")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total = sum(item["bytes"] for item in manifest["files"].values())
    print(f"bundle: {total / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()

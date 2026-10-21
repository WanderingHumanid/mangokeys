"""Assemble the folder the web page is served from.

    python -m scripts.build_web

The page is a folder of static files: the page itself, the runtime of the
browser, the model and the word list. This script copies all of it from the
committed bundle, so the page that GitHub Pages serves and the page that runs
on a laptop are the same files.

The demo folder is not committed, because the model is a binary and the
repository should hold the source. The release workflow builds it and the
Pages workflow publishes it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "apps" / "expo-web"

# The runtime of the browser has to sit next to the page, because the page
# is served from the demo folder and cannot import from the repository.
RUNTIME_FILES = ("transliterator.js", "lexicon.js", "ngram.js", "suggest.js")

# The files of the page itself.
PAGE_FILES = ("index.html", "style.css", "app.js", "worker.js", "serve.py")

# The onnx runtime for the web, copied out of the node modules.
ORT_FILES = (
    "ort.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
)


def copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="assets/release")
    parser.add_argument("--out", default=str(DEMO))
    args = parser.parse_args()

    bundle = ROOT / args.bundle
    out = Path(args.out)
    models = out / "models"

    if not (bundle / "model" / "encoder.onnx").exists():
        raise SystemExit(
            f"no bundle at {bundle}, run python -m scripts.build_release_assets first"
        )

    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    for name in ("encoder.onnx", "decoder.onnx", "tokenizer.json"):
        copy(bundle / "model" / name, models / name)

    # The prefix model stops in the middle of a word, and the word list
    # completes it.
    for name in ("encoder.onnx", "decoder.onnx"):
        copy(bundle / "prefix" / name, models / f"prefix-{name}")

    lexicon = models / "lexicon"
    if lexicon.exists():
        shutil.rmtree(lexicon)
    shutil.copytree(bundle / "lexicon", lexicon)

    for name in RUNTIME_FILES:
        copy(ROOT / "runtime" / "js" / name, out / name)

    for name in PAGE_FILES:
        copy(DEMO / name, out / name)

    # The language model is off by default, because it measured as no help.
    # It is left out of the page so the download is smaller.
    ngram = models / "ngram.txt"
    if ngram.exists():
        ngram.unlink()

    ort_source = ROOT / "node_modules" / "onnxruntime-web" / "dist"
    if not ort_source.exists():
        raise SystemExit("no onnxruntime-web, run npm install first")
    for name in ORT_FILES:
        copy(ort_source / name, out / "ort" / name)

    # The page reads the benchmark table from the demo folder, and it has to
    # be in the page's format rather than the raw evaluation format.
    subprocess.run(
        [sys.executable, "-m", "scripts.make_report", "--demo-out", str(models / "eval.json")],
        check=True,
        cwd=ROOT,
    )

    total = sum(path.stat().st_size for path in models.glob("*.onnx"))
    print(f"the demo is ready in {out}")
    print(f"model payload: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

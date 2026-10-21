"""Build the expo demo assets.

    python -m scripts.build_demo --checkpoint runs/ml_mini/best.pt

Exports the model to ONNX, copies the files into the demo folder and writes
the benchmark table that the page reads. The word list and the language model
are copied as well, because the suggestion strip needs them.
"""

from __future__ import annotations

import argparse

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "apps" / "expo-web" / "models"

RUNTIME_FILES = ("transliterator.js", "lexicon.js", "ngram.js", "suggest.js")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--results", default="evals/results/eval.json")
    parser.add_argument("--lexicon", default="data/processed/lexicon")
    parser.add_argument("--ngram", default="data/processed/ngram.txt")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    onnx_dir = ROOT / "runs" / "demo" / "onnx"
    run(
        [
            sys.executable,
            "-m",
            "model.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(onnx_dir),
        ]
    )

    source_dir = onnx_dir
    if args.quantize:
        quantized = ROOT / "runs" / "demo" / "onnx-int8"
        run(
            [
                sys.executable,
                "-m",
                "model.quantize",
                "--onnx",
                str(onnx_dir),
                "--out",
                str(quantized),
            ]
        )
        source_dir = quantized

    DEMO.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.onnx", "decoder.onnx"):
        shutil.copy(source_dir / name, DEMO / name)
        size = (DEMO / name).stat().st_size / 1e6
        print(f"copied {name} ({size:.1f} MB)")

    shutil.copy(checkpoint.parent / "tokenizer.json", DEMO / "tokenizer.json")
    print("copied tokenizer.json")

    # The runtime has to sit next to the page, because the page is served from
    # the demo folder and cannot import from the repository.
    for name in RUNTIME_FILES:
        shutil.copy(ROOT / "runtime" / "js" / name, DEMO.parent / name)
    print(f"copied {', '.join(RUNTIME_FILES)}")

    # The word list and the language model are what turn the answer of the
    # model into a list of words that exist.
    lexicon = Path(args.lexicon)
    if lexicon.exists():
        target = DEMO / "lexicon"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(lexicon, target)
        size = sum(path.stat().st_size for path in target.iterdir()) / 1e6
        print(f"copied the word list ({size:.1f} MB)")
    else:
        print(f"no word list at {lexicon}, run scripts.build_lexicon first")

    ngram = Path(args.ngram)
    if ngram.exists():
        shutil.copy(ngram, DEMO / "ngram.txt")
        size = ngram.stat().st_size / 1e6
        print(f"copied the language model ({size:.1f} MB)")
    else:
        print(f"no language model at {ngram}, run scripts.train_ngram first")

    # The page reads the benchmark table from the demo folder, and it has to be
    # in the page's format rather than the raw evaluation format.
    run([sys.executable, "-m", "scripts.make_report", "--demo-out", str(DEMO / "eval.json")])

    total = sum(path.stat().st_size for path in DEMO.glob("*.onnx")) / 1e6
    print(f"model payload: {total:.1f} MB")


if __name__ == "__main__":
    main()

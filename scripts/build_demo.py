"""Build the expo demo assets.

    python -m scripts.build_demo --checkpoint runs/ml_mini/best.pt

Exports the model to ONNX, copies the files into the demo folder and writes
the benchmark table that the page reads.
"""

from __future__ import annotations

import argparse

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "apps" / "expo-web" / "models"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--results", default="evals/results/eval.json")
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
    shutil.copy(ROOT / "runtime" / "js" / "transliterator.js", DEMO / "transliterator.js")
    print("copied transliterator.js")

    # The page reads the benchmark table from the demo folder, and it has to be
    # in the page's format rather than the raw evaluation format.
    run([sys.executable, "-m", "scripts.make_report", "--demo-out", str(DEMO / "eval.json")])

    total = sum(path.stat().st_size for path in DEMO.glob("*.onnx")) / 1e6
    print(f"model payload: {total:.1f} MB")


if __name__ == "__main__":
    main()

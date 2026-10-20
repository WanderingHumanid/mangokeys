"""Evaluate both arms of the augmentation ablation.

    python -m scripts.eval_ablation --epochs 12

Trains the two arms on the same number of rows with the same budget, then
evaluates both on the standard test set and on the same words written in the
adhoc style. The arms differ only in what the extra rows look like.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
DATA = ROOT / "data" / "processed"

STANDARD = DATA / "test_aksharantar.tsv"
ADHOC = DATA / "test_aksharantar_adhoc.tsv"


def run(command: list[str]) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT)


def train(name: str, train_file: Path, epochs: int, seed: int) -> None:
    run(
        [
            sys.executable,
            "-m",
            "model.train",
            "--config",
            "configs/ml_mini.yaml",
            "--out-dir",
            f"runs/{name}",
            "--train",
            str(train_file),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
        ]
    )


def evaluate(name: str, checkpoint: Path, device: str) -> None:
    run(
        [
            sys.executable,
            "-m",
            "model.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(STANDARD),
            str(ADHOC),
            "--device",
            device,
            "--out",
            str(RESULTS / f"ablation_{name}.json"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    arms = {
        "e1": DATA / "train_standard_doubled.tsv",
        "e2": DATA / "train.tsv",
    }

    for name, train_file in arms.items():
        if not train_file.exists():
            print(f"skipping {name}, no {train_file}")
            continue
        if not args.skip_training:
            train(f"ablation_{name}", train_file, args.epochs, args.seed)
        checkpoint = ROOT / "runs" / f"ablation_{name}" / "best.pt"
        if not checkpoint.exists():
            print(f"skipping {name}, no checkpoint")
            continue
        evaluate(name, checkpoint, args.device)

    print("\nsummary")
    for name in arms:
        path = RESULTS / f"ablation_{name}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            print(
                f"  {name} {entry['file']:32} top1 {entry['top1']:.4f}  cer {entry['cer']:.4f}"
            )


if __name__ == "__main__":
    main()

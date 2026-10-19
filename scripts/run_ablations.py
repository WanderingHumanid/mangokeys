"""Run the experiment ladder from the plan.

    python -m scripts.run_ablations --stage all

Each stage builds its own data file, trains a model and evaluates it. The
results land in evals/results so that the report can be assembled from one
place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def split_by_style(data_dir: Path) -> None:
    """Separate the standard rows from the augmented ones."""
    run(
        [
            sys.executable,
            "-m",
            "scripts.split_by_style",
            "--data",
            str(data_dir / "train.tsv"),
            "--out",
            str(data_dir),
        ]
    )


def train_and_eval(name: str, data_dir: Path, epochs: int, seed: int) -> dict:
    config = ROOT / "configs" / "ml_mini.yaml"
    run_dir = ROOT / "runs" / name
    run(
        [
            sys.executable,
            "-m",
            "model.train",
            "--config",
            str(config),
            "--out-dir",
            str(run_dir),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "model.evaluate",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data",
            str(data_dir / "test_aksharantar.tsv"),
            "--out",
            str(RESULTS / f"{name}.json"),
        ]
    )
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def train_on(name: str, train_file: Path, data_dir: Path, epochs: int, seed: int) -> dict:
    """Train on one file and evaluate the result on the shared test set."""
    run_dir = ROOT / "runs" / name
    run(
        [
            sys.executable,
            "-m",
            "model.train",
            "--config",
            str(ROOT / "configs" / "ml_mini.yaml"),
            "--out-dir",
            str(run_dir),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--train",
            str(train_file),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "model.evaluate",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data",
            str(data_dir / "test_aksharantar.tsv"),
            "--out",
            str(RESULTS / f"{name}.json"),
        ]
    )
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", help="which ablation to run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        default="data/processed",
        help="the processed directory to run the ablations against",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    main_dir = ROOT / args.data

    if args.stage in ("all", "e1"):
        # The standard rows only, the clean data ceiling.
        split_by_style(main_dir)
        standard = main_dir / "train_standard.tsv"
        if not standard.exists():
            print("no train_standard.tsv, run the data build first")
            return
        summary["e1_standard_only"] = train_on(
            "e1_standard_only", standard, main_dir, args.epochs, args.seed
        )

    if args.stage in ("all", "e2"):
        # Everything, standard and augmented, the main hypothesis.
        summary["e2_with_augment"] = train_on(
            "e2_with_augment", main_dir / "train.tsv", main_dir, args.epochs, args.seed
        )

    if args.stage in ("all", "e6"):
        # Human written Dakshina rows only, against the mined IndicCorp rows.
        for source in ("Dakshina", "IndicCorp"):
            source_file = main_dir / f"train_{source}.tsv"
            if not source_file.exists():
                print(f"skipping {source}, no {source_file}")
                continue
            name = f"e6_{source.lower()}"
            summary[name] = train_on(name, source_file, main_dir, args.epochs, args.seed)

    out = RESULTS / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

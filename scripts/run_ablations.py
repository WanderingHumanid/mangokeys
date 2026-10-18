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

from data.scripts.build_splits import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def build_data(name: str, augment: bool) -> Path:
    """Build a training file for one ablation."""
    out_dir = ROOT / "data" / "processed" / name
    build(
        dakshina=str(ROOT / "data" / "raw" / "dakshina"),
        aksharantar=str(ROOT / "data" / "raw" / "aksharantar_ml"),
        out=str(out_dir),
        augment_copies=1 if augment else 0,
    )
    return out_dir


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", help="which ablation to run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    if args.stage in ("all", "e1"):
        # No augmentation, the clean data ceiling.
        data_dir = build_data("e1_noaugment", augment=False)
        summary["e1_noaugment"] = train_and_eval("e1_noaugment", data_dir, args.epochs, args.seed)

    if args.stage in ("all", "e2"):
        # With the augmentation pipeline, the main hypothesis.
        data_dir = build_data("e2_augment", augment=True)
        summary["e2_augment"] = train_and_eval("e2_augment", data_dir, args.epochs, args.seed)

    if args.stage in ("all", "e6"):
        # Human written Dakshina rows only, against the mined IndicCorp rows.
        # The per source files come out of build_splits.
        for source in ("Dakshina", "IndicCorp"):
            data_dir = ROOT / "data" / "processed" / "e2_augment"
            source_file = data_dir / f"train_{source}.tsv"
            if not source_file.exists():
                print(f"skipping {source}, no {source_file}")
                continue
            run_dir = ROOT / "runs" / f"e6_{source.lower()}"
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
                    str(args.epochs),
                    "--seed",
                    str(args.seed),
                    "--train",
                    str(source_file),
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
                    str(RESULTS / f"e6_{source.lower()}.json"),
                ]
            )
            summary[f"e6_{source.lower()}"] = json.loads(
                (RESULTS / f"e6_{source.lower()}.json").read_text(encoding="utf-8")
            )

    out = RESULTS / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

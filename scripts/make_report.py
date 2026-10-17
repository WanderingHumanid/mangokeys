"""Generate the results section of the report from the evaluation files.

    python -m scripts.make_report

Every number in the report comes from here, so nothing is ever typed in by
hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"

SYSTEM_LABELS = {
    "rules": "rule engine (ours)",
    "indicxlit": "IndicXlit (MIT, 11M)",
    "ours": "our model",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_table(systems: dict[str, list[dict]], test_file: str) -> list[str]:
    lines = [
        "| System | Top-1 | CER | Top-5 | ms / word |",
        "|---|---|---|---|---|",
    ]
    for key, rows in systems.items():
        entry = next((row for row in rows if row.get("file") == test_file), None)
        if entry is None:
            continue
        label = SYSTEM_LABELS.get(key, key)
        lines.append(
            f"| {label} | {percent(entry['top1'])} | {percent(entry['cer'])} | "
            f"{percent(entry.get('top5', 0.0))} | {entry.get('ms_per_pair', 0):.2f} |"
        )
    return lines


def bucket_table(rows: list[dict], test_file: str) -> list[str]:
    entry = next((row for row in rows if row.get("file") == test_file), None)
    if entry is None or not entry.get("buckets"):
        return []

    buckets = entry["buckets"]
    names = list(buckets.keys())
    lines = [
        "| Test set | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for metric, label in (("top1", "Top-1"), ("cer", "CER")):
        values = []
        for name in names:
            summary = buckets[name]
            values.append(percent(summary[metric]))
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    sizes = []
    for name in names:
        sizes.append(str(buckets[name]["n"]))
    lines.append("| words | " + " | ".join(sizes) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default="test_aksharantar.tsv")
    parser.add_argument("--out", default="evals/results/RESULTS.md")
    args = parser.parse_args()

    systems = {
        "rules": load(RESULTS / "baseline_rules.json"),
        "ours": load(RESULTS / "eval.json"),
        "indicxlit": load(RESULTS / "baseline_indicxlit.json"),
    }

    lines = ["## Results", ""]
    lines += build_table(systems, args.test_file)
    lines.append("")

    ours = systems["ours"]
    if ours:
        lines += ["### Breakdown by test subset", ""]
        lines += bucket_table(ours, args.test_file)
        lines.append("")

    ablations = RESULTS / "ablation_summary.json"
    if ablations.exists():
        summary = json.loads(ablations.read_text(encoding="utf-8"))
        lines += ["### Ablations", "", "| Run | Top-1 | CER |", "|---|---|---|"]
        for name, rows in summary.items():
            for row in rows:
                lines.append(
                    f"| {name} | {percent(row['top1'])} | {percent(row['cer'])} |"
                )
        lines.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

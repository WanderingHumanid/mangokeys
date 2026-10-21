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
    "ours_standard_only": "our model, no augmentation",
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


def model_size_mb() -> float:
    """The size of the shipped model, in megabytes."""
    total = 0
    for name in ("encoder.onnx", "decoder.onnx"):
        path = ROOT / "apps" / "expo-web" / "models" / name
        if path.exists():
            total += path.stat().st_size
    return round(total / 1e6, 1)


def build_demo_payload(systems: dict[str, list[dict]], test_file: str) -> list[dict]:
    """Build the file the web demo reads.

    One entry per test file, each with a systems table so that the model, the
    rule baseline and the model trained without augmentation sit side by side.
    """
    files = [test_file]
    adhoc_file = "test_aksharantar_adhoc.tsv"
    payloads = []

    for name, rows in systems.items():
        for entry in rows:
            if entry.get("file") not in files and entry.get("file") != adhoc_file:
                continue
            label = SYSTEM_LABELS.get(name, name)
            size = "3 KB" if name == "rules" else f"{model_size_mb()} MB"
            record = {
                "top1": entry["top1"],
                "cer": entry["cer"],
                "top5": entry.get("top5", 0.0),
                "ms_per_pair": entry.get("ms_per_pair", 0.0),
                "size": size,
            }
            existing = next(
                (p for p in payloads if p["file"] == entry["file"]),
                None,
            )
            if existing is None:
                existing = {"file": entry["file"], "systems": {}}
                payloads.append(existing)
            existing["systems"][label] = record

    # The word list is the layer the keyboard adds on top of the model, so
    # the demo shows it as its own system.
    suggestions = load(RESULTS / "suggestions.json")
    if isinstance(suggestions, dict):
        suggestions = [suggestions]
    if suggestions:
        with_list = suggestions[0].get("with_word_list")
        if with_list:
            for payload in payloads:
                if payload["file"] != test_file:
                    continue
                payload["systems"]["our model with the word list"] = {
                    "top1": with_list["top1"],
                    "cer": with_list["cer"],
                    "top5": with_list.get("top5", 0.0),
                    "ms_per_pair": 0.0,
                    "size": f"{model_size_mb()} MB",
                }
                # The page says how much of the test set the word list knows,
                # because that is the limit of what the keyboard can promise.
                coverage = suggestions[0].get("dictionary_coverage")
                if coverage is not None:
                    payload["dictionary_coverage"] = coverage

    return payloads


ABLATION_LABELS = {
    "ablation_e1": "standard rows only",
    "ablation_e2": "standard + adhoc rows",
}


def build_ablation_table() -> list[str]:
    """Compare the two arms of the augmentation ablation.

    Each arm is evaluated on the standard test set and on the same words
    written in the adhoc style.
    """
    rows: dict[str, dict[str, dict]] = {}
    for key, label in ABLATION_LABELS.items():
        entries = load(RESULTS / f"{key}.json")
        if not entries:
            continue
        rows[label] = {entry["file"]: entry for entry in entries}

    if len(rows) < 2:
        return []

    lines = [
        "| Training data | Standard test set | | Adhoc test set | |",
        "|---|---|---|---|---|",
    ]
    for label, entries in rows.items():
        standard = entries.get("test_aksharantar.tsv")
        adhoc = entries.get("test_aksharantar_adhoc.tsv")
        if standard is None or adhoc is None:
            continue
        lines.append(
            f"| {label} | {percent(standard['top1'])} | {percent(standard['cer'])} "
            f"| {percent(adhoc['top1'])} | {percent(adhoc['cer'])} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default="test_aksharantar.tsv")
    parser.add_argument("--out", default="evals/results/RESULTS.md")
    parser.add_argument("--demo-out", default="apps/expo-web/models/eval.json")
    args = parser.parse_args()

    systems = {
        "rules": load(RESULTS / "baseline_rules.json"),
        "ours": load(RESULTS / "eval.json"),
        "indicxlit": load(RESULTS / "baseline_indicxlit.json"),
    }

    # The augmented model also has numbers on the adhoc test set.
    for entry in load(RESULTS / "ablation_e2.json"):
        if entry.get("file") == "test_aksharantar_adhoc.tsv":
            systems["ours"].append(entry)

    # The standard-trained model, for the adhoc comparison the demo shows.
    ablation_e1 = load(RESULTS / "ablation_e1.json")
    if ablation_e1:
        systems["ours_standard_only"] = ablation_e1

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

    table = build_ablation_table()
    if table:
        lines += ["### Does the augmentation help?", "", *table, ""]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    demo = ROOT / args.demo_out
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_text(
        json.dumps(build_demo_payload(systems, args.test_file), indent=2), encoding="utf-8"
    )
    print(f"wrote {demo}")


if __name__ == "__main__":
    main()

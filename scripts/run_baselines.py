"""Evaluate the rule based baseline on the same test sets as the model.

    python -m scripts.run_baselines --data data/processed/test_aksharantar.tsv

The numbers come out in the same shape as the model evaluation so that the two
can be put in one table.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.src.metrics import ErrorCounter  # noqa: E402
from model.src.rules import RuleEngine  # noqa: E402


def read_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row = {"roman": parts[0], "native": parts[1]}
            if len(parts) > 2 and parts[2]:
                row["bucket"] = parts[2]
            rows.append(row)
    return rows


def evaluate(rows: list[dict]) -> dict:
    engine = RuleEngine()
    overall = ErrorCounter()
    per_bucket: dict[str, ErrorCounter] = defaultdict(ErrorCounter)

    start = time.time()
    for row in rows:
        candidates = engine.convert(row["roman"])
        best = candidates[0] if candidates else ""
        overall.update(row["native"], best)
        overall.update_ranking(row["native"], candidates)
        per_bucket[row.get("bucket", "all")].update(row["native"], best)
    elapsed = time.time() - start

    result = {
        "n": len(rows),
        "top1": round(overall.accuracy, 4),
        "cer": round(overall.cer, 4),
        "top5": round(overall.recall_at_k, 4),
        "seconds": round(elapsed, 1),
        "ms_per_pair": round(elapsed * 1000 / max(len(rows), 1), 3),
        "buckets": {name: counter.summary() for name, counter in sorted(per_bucket.items())},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["data/processed/test_aksharantar.tsv"])
    parser.add_argument("--out", default="evals/results/baseline_rules.json")
    args = parser.parse_args()

    results = []
    for data_path in args.data:
        path = Path(data_path)
        if not path.exists():
            print(f"skipping missing {path}")
            continue
        rows = read_pairs(path)
        result = evaluate(rows)
        result["file"] = path.name
        results.append(result)
        print(
            f"{path.name}: n={result['n']} top1={result['top1']:.4f} "
            f"cer={result['cer']:.4f} top5={result['top5']:.4f} "
            f"({result['ms_per_pair']} ms/pair)"
        )
        for name, summary in result["buckets"].items():
            print(f"    {name:12} n={summary['n']:6} top1={summary['top1']:.4f} cer={summary['cer']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

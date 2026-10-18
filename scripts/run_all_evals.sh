#!/usr/bin/env bash
# Run every evaluation and rebuild the report.
#
#   ./scripts/run_all_evals.sh [checkpoint]
#
# The checkpoint defaults to runs/ml_mini/best.pt.

set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-runs/ml_mini/best.pt}"
TEST_DATA="data/processed/test_aksharantar.tsv"

echo "=== rule engine baseline ==="
python -m scripts.run_baselines --data "$TEST_DATA"

if [ -f "$CHECKPOINT" ]; then
    echo
    echo "=== model: $CHECKPOINT ==="
    python -m model.evaluate --checkpoint "$CHECKPOINT" --data "$TEST_DATA" \
        --out evals/results/eval.json
else
    echo
    echo "no checkpoint at $CHECKPOINT, skipping the model evaluation"
fi

echo
echo "=== report ==="
python -m scripts.make_report

"""Check that the ONNX runtime and the PyTorch model agree.

    python -m scripts.check_parity --checkpoint runs/ml_mini/best.pt --onnx runs/ml_mini/onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.evaluate import load_model  # noqa: E402
from model.src.decode import beam_search  # noqa: E402
from model.src.data import Collator, PairDataset  # noqa: E402
from runtime.python.transliterator import Transliterator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--onnx", default="runs/ml_mini/onnx")
    parser.add_argument("--data", default="data/processed/valid.tsv")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer, config = load_model(Path(args.checkpoint), device)

    engine = Transliterator(
        args.onnx,
        Path(args.checkpoint).parent / "tokenizer.json",
        max_src=config["data"]["max_src"],
        max_tgt=config["data"]["max_tgt"],
        beam=4,
    )

    rows: list[dict] = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append({"roman": parts[0], "native": parts[1]})
            if len(rows) >= args.limit:
                break

    dataset = PairDataset(rows, tokenizer, config["data"]["max_src"], config["data"]["max_tgt"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=Collator(tokenizer))

    torch_results: list[list[str]] = []
    for batch in loader:
        candidates = beam_search(
            model,
            batch["src"],
            batch["src_mask"],
            bos_id=tokenizer.tgt.bos_id,
            eos_id=tokenizer.tgt.eos_id,
            beam=4,
            max_len=config["data"]["max_tgt"],
            n_best=5,
        )
        for group in candidates:
            torch_results.append([tokenizer.decode_tgt(ids) for ids in group])

    onnx_results = [engine.candidates(row["roman"], n_best=5) for row in rows]

    mismatches = 0
    for index, (expected, actual) in enumerate(zip(torch_results, onnx_results, strict=True)):
        if expected[:1] != actual[:1]:
            mismatches += 1
            if mismatches <= 5:
                print(
                f"row {index}: torch={expected[:1]} onnx={actual[:1]} "
                f"roman={rows[index]['roman']!r}"
            )

    total = len(rows)
    print(f"checked {total} rows, {mismatches} mismatches")
    if mismatches:
        print("parity check failed")
        raise SystemExit(1)
    print("parity ok")


if __name__ == "__main__":
    main()

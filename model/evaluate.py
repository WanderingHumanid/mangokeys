"""Evaluate a trained checkpoint.

    python -m model.evaluate --checkpoint runs/ml_mini/best.pt --data data/processed/test_aksharantar.tsv

Reports top-1 accuracy, character error rate and top-5 recall for every test
set, broken down by the bucket label when the file has one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.src.data import Collator, PairDataset
from model.src.decode import beam_search, greedy_decode
from model.src.metrics import ErrorCounter
from model.src.seq2seq import Seq2Seq
from model.src.tokenizer import TranslitTokenizer


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Seq2Seq, TranslitTokenizer, dict]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    tokenizer = TranslitTokenizer.load(checkpoint_path.parent / "tokenizer.json")

    settings = config["model"]
    model = Seq2Seq(
        src_vocab=tokenizer.src_vocab,
        tgt_vocab=tokenizer.tgt_vocab,
        d_model=settings["d_model"],
        n_heads=settings["n_heads"],
        d_ff=settings["d_ff"],
        n_enc_layers=settings["n_enc_layers"],
        n_dec_layers=settings["n_dec_layers"],
        dropout=0.0,
        max_len=max(config["data"]["max_src"], config["data"]["max_tgt"]) + 8,
        pad_id=tokenizer.src.pad_id,
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, tokenizer, config


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


def evaluate_file(
    model: Seq2Seq,
    tokenizer: TranslitTokenizer,
    rows: list[dict],
    config: dict,
    device: torch.device,
    beam: int,
    n_best: int,
    batch_size: int,
) -> dict:
    dataset = PairDataset(rows, tokenizer, config["data"]["max_src"], config["data"]["max_tgt"])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(tokenizer))

    overall = ErrorCounter()
    per_bucket: dict[str, ErrorCounter] = defaultdict(ErrorCounter)
    references: list[str] = []
    hypotheses: list[str] = []

    start = time.time()
    offset = 0
    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)

        if beam > 1:
            candidates = beam_search(
                model,
                src,
                src_mask,
                bos_id=tokenizer.tgt.bos_id,
                eos_id=tokenizer.tgt.eos_id,
                beam=beam,
                max_len=config["data"]["max_tgt"],
                n_best=n_best,
            )
        else:
            single = greedy_decode(
                model,
                src,
                src_mask,
                bos_id=tokenizer.tgt.bos_id,
                eos_id=tokenizer.tgt.eos_id,
                max_len=config["data"]["max_tgt"],
            )
            candidates = [[ids] for ids in single]

        for row, batch_candidates in enumerate(candidates):
            reference = tokenizer.decode_tgt(batch["tgt_out"][row].tolist())
            decoded = [tokenizer.decode_tgt(ids) for ids in batch_candidates]
            best = decoded[0] if decoded else ""
            overall.update(reference, best)
            overall.update_ranking(reference, decoded)
            bucket = rows[offset + row].get("bucket", "all")
            per_bucket[bucket].update(reference, best)
            references.append(reference)
            hypotheses.append(best)
        offset += len(candidates)

    elapsed = time.time() - start
    total = max(len(rows), 1)

    result = {
        "file": "unknown",
        "n": len(rows),
        "top1": round(overall.accuracy, 4),
        "cer": round(overall.cer, 4),
        "top5": round(overall.recall_at_k, 4),
        "seconds": round(elapsed, 1),
        "ms_per_pair": round(elapsed * 1000 / total, 2),
        "buckets": {},
    }
    for name, counter in sorted(per_bucket.items()):
        result["buckets"][name] = counter.summary()

    return result


def evaluate_onnx(
    onnx_dir: Path,
    tokenizer_path: Path,
    rows: list[dict],
    config: dict,
    device: torch.device,
    beam: int,
    n_best: int,
    batch_size: int,
) -> dict:
    """Evaluate an exported model instead of the pytorch checkpoint."""
    from runtime.python.transliterator import Transliterator

    engine = Transliterator(
        onnx_dir,
        tokenizer_path,
        max_src=config["data"]["max_src"],
        max_tgt=config["data"]["max_tgt"],
        beam=beam,
    )
    overall = ErrorCounter()
    per_bucket: dict[str, ErrorCounter] = defaultdict(ErrorCounter)

    start = time.time()
    for row in rows:
        candidates = engine.candidates(row["roman"], n_best=n_best)
        reference = row["native"]
        best = candidates[0] if candidates else ""
        overall.update(reference, best)
        overall.update_ranking(reference, candidates)
        per_bucket[row.get("bucket", "all")].update(reference, best)
    elapsed = time.time() - start

    return {
        "file": "unknown",
        "n": len(rows),
        "top1": round(overall.accuracy, 4),
        "cer": round(overall.cer, 4),
        "top5": round(overall.recall_at_k, 4),
        "seconds": round(elapsed, 1),
        "ms_per_pair": round(elapsed * 1000 / max(len(rows), 1), 2),
        "buckets": {name: counter.summary() for name, counter in sorted(per_bucket.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--onnx", default=None, help="evaluate an exported model instead")
    parser.add_argument("--data", nargs="+", default=["data/processed/test_akshantar.tsv"])
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=None, help="write the results as json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    model, tokenizer, config = load_model(checkpoint, device)

    results = []
    for data_path in args.data:
        path = Path(data_path)
        if not path.exists():
            print(f"skipping missing {path}")
            continue
        rows = read_pairs(path)
        if args.onnx:
            result = evaluate_onnx(
                Path(args.onnx),
                checkpoint.parent / "tokenizer.json",
                rows,
                config,
                device,
                beam=args.beam,
                n_best=args.n_best,
                batch_size=args.batch_size,
            )
        else:
            result = evaluate_file(
                model,
                tokenizer,
                rows,
                config,
                device,
                beam=args.beam,
                n_best=args.n_best,
                batch_size=args.batch_size,
            )
        result["file"] = path.name
        results.append(result)
        print(
            f"{path.name}: n={result['n']} top1={result['top1']:.4f} "
            f"cer={result['cer']:.4f} top5={result['top5']:.4f} "
            f"({result['ms_per_pair']} ms/pair)"
        )
        for name, summary in result["buckets"].items():
            print(f"    {name:12} n={summary['n']:6} top1={summary['top1']:.4f} cer={summary['cer']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

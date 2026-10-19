"""Dump and classify the errors the model makes.

    python -m scripts.error_analysis --checkpoint runs/ml_mini/best.pt \
        --data data/processed/test_aksharantar.tsv

Writes the worst errors to a tsv and counts them by class. The classes are
the ones in docs/EVAL_PROTOCOL.md. A class that takes more than a fifth of the
errors is a signal that a data rule needs to change.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

from model.evaluate import load_model, read_pairs
from model.src.decode import beam_search
from model.src.data import Collator, PairDataset


LONG_VOWEL_SIGNS = "ാീൂോൗ"
SHORT_VOWEL_SIGNS = "ിുെൊൌ"


def classify(reference: str, hypothesis: str, roman: str) -> str:
    """Guess which failure mode produced an error."""
    ref = reference
    hyp = hypothesis

    # A missing long vowel sign is the most common failure.
    missing_long = sum(ref.count(sign) for sign in LONG_VOWEL_SIGNS) - sum(
        hyp.count(sign) for sign in LONG_VOWEL_SIGNS
    )
    if missing_long > 0:
        return "missing-long-vowel"

    missing_virama = ref.count("്") - hyp.count("്")
    if missing_virama > 0:
        return "missing-virama"

    extra_virama = hyp.count("്") - ref.count("്")
    if extra_virama > 0:
        return "extra-virama"

    missing_short = sum(ref.count(sign) for sign in SHORT_VOWEL_SIGNS) - sum(
        hyp.count(sign) for sign in SHORT_VOWEL_SIGNS
    )
    if missing_short > 0:
        return "missing-short-vowel"

    if any(ch in "ൺൻർൽൾൿ" for ch in ref) and not any(
        ch in "ൺൻർൽൾൿ" for ch in hyp
    ):
        return "chillu"
    if "ശ" in ref and "ശ" not in hyp and "ഷ" in hyp:
        return "sh-vs-sha"
    if "ഷ" in ref and "ഷ" not in hyp and "ശ" in hyp:
        return "sh-vs-sha"
    if "സ" in ref and "സ" not in hyp and "ശ" in hyp:
        return "s-vs-sh"
    if "ട" in ref and "ട" not in hyp and "ത" in hyp:
        return "retroflex"
    if "ത" in ref and "ത" not in hyp and "ട" in hyp:
        return "retroflex"
    if "ഴ" in ref and "ഴ" not in hyp:
        return "zha"
    if "ള" in ref and "ള" not in hyp and "ല" in hyp:
        return "lla-vs-la"
    if "ണ" in ref and "ണ" not in hyp and "ന" in hyp:
        return "nna-vs-na"
    if "റ" in ref and "റ" not in hyp and "ര" in hyp:
        return "rra-vs-ra"
    if "ന്ത" in ref and "ന്ത" not in hyp and "ന്റ" in hyp:
        return "cluster"
    if "ണ്ട" in ref and "ണ്ട" not in hyp and "ന്ദ" in hyp:
        return "cluster"
    if any(ch.isascii() and ch.isalpha() for ch in hyp):
        return "latin-leak"
    if len(hyp) < len(ref):
        return "too-short"
    if len(hyp) > len(ref):
        return "too-long"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--data", default="data/processed/test_aksharantar.tsv")
    parser.add_argument("--out", default="evals/results/errors.tsv")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, tokenizer, config = load_model(Path(args.checkpoint), device)

    rows = read_pairs(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    dataset = PairDataset(rows, tokenizer, config["data"]["max_src"], config["data"]["max_tgt"])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=Collator(tokenizer)
    )

    errors: list[dict] = []
    classes: collections.Counter[str] = collections.Counter()
    correct = 0
    total = 0
    offset = 0

    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        candidates = beam_search(
            model,
            src,
            src_mask,
            bos_id=tokenizer.tgt.bos_id,
            eos_id=tokenizer.tgt.eos_id,
            beam=args.beam,
            max_len=config["data"]["max_tgt"],
            n_best=1,
        )
        for row, group in enumerate(candidates):
            hypothesis = tokenizer.decode_tgt(group[0]) if group else ""
            reference = tokenizer.decode_tgt(batch["tgt_out"][row].tolist())
            total += 1
            if hypothesis == reference:
                correct += 1
                continue
            label = classify(reference, hypothesis, rows[offset + row]["roman"])
            classes[label] += 1
            errors.append(
                {
                    "roman": rows[offset + row]["roman"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "class": label,
                }
            )
        offset += len(candidates)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("roman\treference\thypothesis\tclass\n")
        for error in sorted(errors, key=lambda e: e["roman"]):
            handle.write(
                f"{error['roman']}\t{error['reference']}\t{error['hypothesis']}\t{error['class']}\n"
            )

    print(f"n={total}  correct={correct}  errors={len(errors)}")
    print(f"top-1 {correct / max(total, 1):.4f}\n")
    print("error classes")
    print("-" * 40)
    for name, count in classes.most_common():
        share = count * 100 / max(len(errors), 1)
        flag = "  <-- needs a data rule" if share > 20 else ""
        print(f"  {name:16} {count:6}  {share:5.1f}%{flag}")

    summary = {
        "n": total,
        "correct": correct,
        "errors": len(errors),
        "top1": round(correct / max(total, 1), 4),
        "classes": dict(classes.most_common()),
    }
    Path("evals/results/error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

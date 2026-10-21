"""Train the transliteration model.

    python -m model.train --config configs/ml_mini.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from model.src.data import BucketBatchSampler, Collator, PairDataset
from model.src.decode import greedy_decode
from model.src.metrics import ErrorCounter
from model.src.seq2seq import LabelSmoothedLoss, Seq2Seq
from model.src.tokenizer import CharTokenizer, TranslitTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(path: str) -> list[tuple[str, str]]:
    """Read pairs as plain tuples. Millions of dicts would not fit in memory."""
    rows: list[tuple[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows


def build_tokenizer(pairs: list) -> TranslitTokenizer:
    src = CharTokenizer.build_source()
    tgt = CharTokenizer.build_target([native for _, native in pairs])
    return TranslitTokenizer(src=src, tgt=tgt)


def build_tokenizer_from_file(path: str) -> TranslitTokenizer:
    """Build the tokenizer by reading the file once, without keeping it."""
    src = CharTokenizer.build_source()
    tgt = CharTokenizer.build_target(_natives(path))
    return TranslitTokenizer(src=src, tgt=tgt)


def _natives(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            yield parts[1]


def build_model(config: dict, tokenizer: TranslitTokenizer) -> Seq2Seq:
    settings = config["model"]
    return Seq2Seq(
        src_vocab=tokenizer.src_vocab,
        tgt_vocab=tokenizer.tgt_vocab,
        d_model=settings["d_model"],
        n_heads=settings["n_heads"],
        d_ff=settings["d_ff"],
        n_enc_layers=settings["n_enc_layers"],
        n_dec_layers=settings["n_dec_layers"],
        dropout=settings["dropout"],
        max_len=max(config["data"]["max_src"], config["data"]["max_tgt"]) + 8,
        pad_id=tokenizer.src.pad_id,
    )


def inverse_sqrt_lr(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    return peak * math.sqrt(warmup / max(step, 1))


def evaluate(
    model: Seq2Seq,
    loader: DataLoader,
    tokenizer: TranslitTokenizer,
    max_tgt: int,
) -> dict:
    counter = ErrorCounter()
    model.eval()
    device = next(model.parameters()).device
    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        targets = batch["tgt_out"]
        outputs = greedy_decode(
            model,
            src,
            src_mask,
            bos_id=tokenizer.tgt.bos_id,
            eos_id=tokenizer.tgt.eos_id,
            max_len=max_tgt,
        )
        for row, ids in enumerate(outputs):
            hypothesis = tokenizer.decode_tgt(ids)
            reference = tokenizer.decode_tgt(targets[row].tolist())
            counter.update(reference, hypothesis)
    # The training monitor only does greedy decoding, so there is no ranking
    # metric to report. The full evaluation reports top-5.
    summary = counter.summary()
    summary.pop("top5", None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml_mini.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--out-dir", default=None, help="override the output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train", default=None, help="override the training file")
    parser.add_argument("--valid", default=None, help="override the validation file")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--tokenizer", default=None, help="load the tokenizer from a file")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = config["train"]
    max_tgt = config["data"]["max_tgt"]
    if args.out_dir:
        settings["out_dir"] = args.out_dir
    if args.epochs:
        settings["epochs"] = args.epochs
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.train:
        config["data"]["train"] = args.train
    if args.valid:
        config["data"]["valid"] = args.valid
    settings["amp"] = torch.cuda.is_available() and not args.no_amp
    set_seed(settings["seed"])

    out_dir = Path(settings["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    valid_pairs = load_pairs(config["data"]["valid"])
    print(f"valid pairs: {len(valid_pairs)}")

    if args.tokenizer:
        tokenizer = TranslitTokenizer.load(args.tokenizer)
        print(f"tokenizer loaded from {args.tokenizer}")
    else:
        tokenizer = build_tokenizer_from_file(config["data"]["train"])
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"vocab: source {tokenizer.src_vocab}  target {tokenizer.tgt_vocab}")

    max_src = config["data"]["max_src"]
    # The training file is read from disk, because a file with tens of
    # millions of rows does not fit in memory as a list of strings.
    train_set = PairDataset.from_file(
        config["data"]["train"], tokenizer, max_src, max_tgt
    )
    valid_set = PairDataset.from_file(
        config["data"]["valid"], tokenizer, max_src, max_tgt
    )
    del valid_pairs
    collator = Collator(tokenizer)

    train_sampler = BucketBatchSampler(
        train_set.lengths, settings["batch_size"], seed=settings["seed"]
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=settings["num_workers"],
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=256,
        collate_fn=collator,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params / 1e6:.2f}M   device: {device}   amp: {settings['amp']}")

    criterion = LabelSmoothedLoss(
        vocab_size=tokenizer.tgt_vocab,
        pad_id=tokenizer.tgt.pad_id,
        smoothing=settings["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], betas=(0.9, 0.98), weight_decay=0.01
    )

    start_epoch = 0
    best_cer = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_cer = state.get("best_cer", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    step = start_epoch * len(train_loader)
    history: list[dict] = []

    for epoch in range(start_epoch, settings["epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        start = time.time()
        scaler = torch.amp.GradScaler(device.type, enabled=settings["amp"])

        for index, batch in enumerate(train_loader):
            lr = inverse_sqrt_lr(step, settings["warmup_steps"], settings["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            src = batch["src"].to(device, non_blocking=True)
            src_mask = batch["src_mask"].to(device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(device, non_blocking=True)

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=settings["amp"]):
                logits = model(src, tgt_in, src_mask)
                loss = criterion(logits, tgt_out)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            step += 1

            if index % settings["log_every"] == 0:
                print(
                    f"epoch {epoch} step {step} loss {running / (index + 1):.4f} "
                    f"lr {lr:.2e} {time.time() - start:.0f}s",
                    flush=True,
                )

        if (epoch + 1) % settings["eval_every"] == 0:
            metrics = evaluate(model, valid_loader, tokenizer, max_tgt)
            metrics["epoch"] = epoch
            metrics["train_loss"] = running / max(len(train_loader), 1)
            history.append(metrics)
            print(f"valid  {json.dumps(metrics)}", flush=True)

            if metrics["cer"] < best_cer:
                best_cer = metrics["cer"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_cer": best_cer,
                        "config": config,
                    },
                    out_dir / "best.pt",
                )
                print(f"saved best checkpoint, cer {best_cer:.4f}", flush=True)

            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": settings["epochs"] - 1,
            "best_cer": best_cer,
            "config": config,
        },
        out_dir / "last.pt",
    )
    print("training finished")


if __name__ == "__main__":
    main()

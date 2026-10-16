"""Export a trained checkpoint to ONNX.

    python -m model.export_onnx --checkpoint runs/ml_mini/best.pt --out runs/ml_mini/onnx

The encoder and the decoder step are exported as two separate graphs. The
decoder graph only holds the decoder weights, so the two files together are the
same size as the model and not twice that. The runtime encodes once and then
steps the decoder in a loop, which is what makes beam search possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model.src.seq2seq import Seq2Seq


class EncoderGraph(nn.Module):
    """Turns padded source ids into the decoder memory."""

    def __init__(self, model: Seq2Seq) -> None:
        super().__init__()
        self.model = model

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode(src_ids, src_mask)


class DecoderGraph(nn.Module):
    """One decoder step. Only the decoder side of the model is traced.

    The causal mask comes in as an input. A mask built from a length that is
    only known at run time would be frozen when the graph is traced.
    """

    def __init__(self, model: Seq2Seq) -> None:
        super().__init__()
        self.tgt_embed = model.tgt_embed
        self.pos = model.pos
        self.decoder = model.decoder
        self.dec_norm = model.dec_norm
        self.output_proj = model.output_proj
        self.dropout = model.dropout

    def forward(
        self,
        tgt_prefix: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        self_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt_prefix)
        x = self.pos(x)
        cross_mask = (src_mask == 0).view(src_mask.size(0), 1, 1, src_mask.size(1))
        for layer in self.decoder:
            x = layer(x, memory, self_mask, cross_mask)
        return self.output_proj(self.dec_norm(x))[:, -1]


def load_model(checkpoint: Path, device: torch.device) -> tuple[Seq2Seq, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    from model.src.tokenizer import TranslitTokenizer

    tokenizer = TranslitTokenizer.load(checkpoint.parent / "tokenizer.json")
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
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ml_mini/best.pt")
    parser.add_argument("--out", default="runs/ml_mini/onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-src", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint = Path(args.checkpoint)
    model, config = load_model(checkpoint, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_len = config["data"]["max_src"]
    sample_src = torch.randint(3, 90, (1, src_len), dtype=torch.long)
    sample_mask = torch.ones(1, src_len, dtype=torch.bool)
    sample_tgt = torch.zeros(1, 1, dtype=torch.long)
    sample_self_mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    # The batch axis stays dynamic so that a beam can be decoded in one call.
    # The decoder step axis is dynamic too, since the runtime grows the target
    # prefix as the search proceeds.
    encoder_axes = {"src_ids": {0: "batch"}, "src_mask": {0: "batch"}, "memory": {0: "batch"}}
    decoder_axes = {
        "tgt_prefix": {0: "batch", 1: "step"},
        "memory": {0: "batch"},
        "src_mask": {0: "batch"},
        "self_mask": {2: "step", 3: "step"},
        "logits": {0: "batch"},
    }

    encoder = EncoderGraph(model).eval()
    memory = encoder(sample_src, sample_mask)

    encoder_path = out_dir / "encoder.onnx"
    torch.onnx.export(
        encoder,
        (sample_src, sample_mask),
        encoder_path,
        input_names=["src_ids", "src_mask"],
        output_names=["memory"],
        dynamic_axes=encoder_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    decoder = DecoderGraph(model).eval()
    decoder_path = out_dir / "decoder.onnx"
    torch.onnx.export(
        decoder,
        (sample_tgt, memory, sample_mask, sample_self_mask),
        decoder_path,
        input_names=["tgt_prefix", "memory", "src_mask", "self_mask"],
        output_names=["logits"],
        dynamic_axes=decoder_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    for path in (encoder_path, decoder_path):
        size = path.stat().st_size / 1e6
        print(f"wrote {path} ({size:.1f} MB)")


if __name__ == "__main__":
    main()

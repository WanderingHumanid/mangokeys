"""Quantize the exported ONNX graphs to int8.

    python -m model.quantize --onnx runs/ml_mini/onnx --out runs/ml_mini/onnx-int8

Weight only dynamic quantization keeps the graph shape agnostic, which matters
because the decoder prefix length changes at run time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="runs/ml_mini/onnx")
    parser.add_argument("--out", default="runs/ml_mini/onnx-int8")
    parser.add_argument("--op-types", nargs="+", default=["MatMul", "Gemm"])
    args = parser.parse_args()

    source = Path(args.onnx)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)

    for name in ("encoder.onnx", "decoder.onnx"):
        input_path = source / name
        output_path = target / name
        print(f"quantizing {input_path}")
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=args.op_types,
            extra_options={"MatMulConstBOnly": True},
        )
        before = input_path.stat().st_size / 1e6
        after = output_path.stat().st_size / 1e6
        print(f"  {before:.1f} MB -> {after:.1f} MB")


if __name__ == "__main__":
    main()

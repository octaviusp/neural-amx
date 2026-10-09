"""neural-amx command line.

    neural-amx convert model.onnx -o policy.safetensors --precision int4 --labels attack,hold,retreat
    neural-amx convert weights.safetensors --spec "input(64) dense(fc1) relu dense(fc2)" -o policy.safetensors
    neural-amx convert model.keras -o policy.safetensors          # TensorFlow / Keras 3
    neural-amx convert model.pt -o policy.safetensors              # torch.save(model) of a whole module
    neural-amx info policy.safetensors
    neural-amx run policy.safetensors 0.1,0.5,... [--precision int4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import format as fmt
from .quantize import quantize
from .reference import Runner


def _convert(args):
    src = Path(args.source)
    ext = src.suffix.lower()
    labels = args.labels.split(",") if args.labels else None
    if ext == ".safetensors":
        _, meta = fmt.read_safetensors(src)
        if args.spec:
            from .convert.spec import from_safetensors
            model = from_safetensors(src, args.spec, labels)
        elif "neural_amx" in meta:
            model = fmt.load(src)
        else:
            sys.exit("this safetensors file has no neural-amx graph: pass --spec \"input(N) dense(fc1) relu ...\"")
    elif ext == ".onnx":
        from .convert.onnx import from_onnx
        model = from_onnx(str(src), labels)
    elif ext in (".keras", ".h5"):
        import keras
        from .convert.keras import from_keras
        model = from_keras(keras.models.load_model(src), labels)
    elif ext in (".pt", ".pth"):
        import torch
        from .convert.torch import from_torch
        model = from_torch(torch.load(src, weights_only=False).eval(), args.inputs, labels)
    else:
        sys.exit(f"unsupported source {ext}: use .safetensors (+--spec), .onnx, .keras/.h5 or .pt (MLX: Python API)")
    if args.inputs:
        model.inputs = args.inputs
    if labels:
        model.labels = labels
    if args.precision in ("int8", "int4"):
        model = quantize(model, args.precision)
    Runner(model)  # validates shapes before writing
    out = fmt.save(model, args.output)
    print(f"wrote {out} ({out.stat().st_size} bytes): {_summary(model)}")


def _summary(model) -> str:
    r = Runner(model)
    ops = " ".join(l["op"] + (f"[int{l['bits']}]" if l.get("bits") else "") for l in model.layers)
    return f"{r.inputs} -> {r.outputs}: {ops}"


def _info(args):
    model = fmt.load(args.file)
    print(_summary(model))
    if model.labels:
        print("labels:", ", ".join(model.labels))
    for name, t in model.tensors.items():
        print(f"  {name:32} {str(t.dtype):8} {list(t.shape)}")


def _run(args):
    model = fmt.load(args.file)
    x = np.array([float(v) for v in args.values.split(",")], np.float32)[None, :]
    y = Runner(model, args.precision).run(x)[0]
    best = int(np.argmax(y))
    label = f" ({model.labels[best]})" if model.labels and best < len(model.labels) else ""
    print(" ".join(f"{v:.6g}" for v in y))
    print(f"argmax {best}{label}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="neural-amx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert a model to a neural-amx safetensors file")
    c.add_argument("source")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--spec", help="layer list for a plain state_dict")
    c.add_argument("--inputs", type=int, help="input size (conv models)")
    c.add_argument("--precision", choices=["stored", "int8", "int4"], default="stored", help="store weights quantized")
    c.add_argument("--labels", help="comma-separated output names")
    c.set_defaults(fn=_convert)
    i = sub.add_parser("info", help="describe a model")
    i.add_argument("file")
    i.set_defaults(fn=_info)
    r = sub.add_parser("run", help="run the reference implementation on one input")
    r.add_argument("file")
    r.add_argument("values", help="comma-separated input values")
    r.add_argument("--precision", choices=["stored", "f32", "int8", "int4"], default="stored")
    r.set_defaults(fn=_run)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

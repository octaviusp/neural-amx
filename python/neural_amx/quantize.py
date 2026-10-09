"""Offline weight quantization: stores int8 (I8) or packed int4 (U8, two per byte, low nibble first) weights plus a
per-row float32 "<weight>.scale" tensor. Uses the exact rule of the C loader, so loading a quantized file gives the
same bits as loading the float file with the same precision.
"""
from __future__ import annotations

import copy

import numpy as np

from .format import Model
from .reference import Runner, quantize_weights


def pack_int4(q: np.ndarray) -> np.ndarray:
    rows, cols = q.shape
    padded = np.zeros((rows, cols + (cols & 1)), np.int64)
    padded[:, :cols] = q
    lo, hi = padded[:, 0::2] & 0xF, padded[:, 1::2] & 0xF
    return (lo | (hi << 4)).astype(np.uint8)


def quantize(model: Model, precision: str = "int4") -> Model:
    """Returns a copy whose dense/conv1d/gru weights are stored quantized. precision: "int8" or "int4"
    (int4 keeps the first and last weighted layer and recurrent weights at int8)."""
    if precision not in ("int8", "int4"):
        raise ValueError("precision must be 'int8' or 'int4'")
    out = Model(copy.deepcopy(model.layers), dict(model.tensors), model.inputs, model.labels, dict(model.extra))
    ops = [l for l in out.layers if l["op"] in ("dense", "linear", "conv1d", "gru")]
    for k, layer in enumerate(ops):
        bits = Runner._target_bits(precision, k, len(ops), layer["op"] == "gru")
        names = ["weight_ih", "weight_hh"] if layer["op"] == "gru" else ["weight"]
        for key in names:
            name = layer[key]
            w = out.tensors[name]
            if w.dtype.kind != "f":
                continue  # already quantized
            if layer["op"] == "conv1d" and w.ndim == 3:
                layer.setdefault("in_channels", int(w.shape[1]))
                layer.setdefault("kernel_size", int(w.shape[2]))
            w2 = w.astype(np.float32).reshape(w.shape[0], -1)
            q, s = quantize_weights(w2, bits)
            out.tensors[name] = q.astype(np.int8) if bits == 8 else pack_int4(q)
            out.tensors[name + ".scale"] = s.astype(np.float32)
        layer["bits"] = bits
    return out

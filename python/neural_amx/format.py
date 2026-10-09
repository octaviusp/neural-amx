"""The neural-amx model format: a standard safetensors file whose metadata key "neural_amx" holds the graph as JSON.

Any safetensors reader can open it; the graph lists layers in order and names their tensors.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
WEIGHTED_OPS = ("dense", "conv1d", "gru")
ACTIVATIONS = ("relu", "leaky_relu", "sigmoid", "tanh", "gelu", "gelu_tanh", "silu", "softmax")
OPS = WEIGHTED_OPS + ("layernorm", "affine", "batchnorm", "transpose", "flatten", "identity") + ACTIVATIONS

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I8": np.int8, "U8": np.uint8, "I16": np.int16,
           "I32": np.int32, "I64": np.int64, "BOOL": np.bool_}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}


@dataclass
class Model:
    """A sequential network: `layers` are graph dicts ({"op": ..., tensor names, options}), `tensors` the arrays."""

    layers: list[dict]
    tensors: dict[str, np.ndarray]
    inputs: int | None = None
    labels: list[str] | None = None
    extra: dict = field(default_factory=dict)

    def graph(self) -> dict:
        g = {"format": FORMAT_VERSION, "layers": self.layers}
        if self.inputs:
            g["inputs"] = int(self.inputs)
        if self.labels:
            g["labels"] = list(self.labels)
        g.update(self.extra)
        return g

    def save(self, path: str | Path) -> Path:
        return save(self, path)


def save(model: Model, path: str | Path) -> Path:
    for i, layer in enumerate(model.layers):
        if layer.get("op") not in OPS:
            raise ValueError(f"layer {i}: unknown op {layer.get('op')!r}")
    write_safetensors(path, model.tensors, {"neural_amx": json.dumps(model.graph(), separators=(",", ":"))})
    return Path(path)


def load(path: str | Path) -> Model:
    tensors, meta = read_safetensors(path)
    if "neural_amx" not in meta:
        raise ValueError(f"{path}: no 'neural_amx' graph in the metadata (use from_safetensors with a spec)")
    g = json.loads(meta["neural_amx"])
    extra = {k: v for k, v in g.items() if k not in ("format", "layers", "inputs", "labels")}
    return Model(g["layers"], tensors, g.get("inputs"), g.get("labels"), extra)


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _NAMES:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        data = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": _NAMES[arr.dtype], "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def read_safetensors(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    buf = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", buf[:8])
    header = json.loads(buf[8:8 + n])
    meta = header.pop("__metadata__", {}) or {}
    data = memoryview(buf)[8 + n:]
    tensors = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        if info["dtype"] == "BF16":
            u = np.frombuffer(data[b:e], dtype="<u2").astype(np.uint32) << 16
            arr = u.view(np.float32)
        else:
            arr = np.frombuffer(data[b:e], dtype=np.dtype(_DTYPES[info["dtype"]]).newbyteorder("<"))
        tensors[name] = arr.reshape(info["shape"]).copy()
    return tensors, meta

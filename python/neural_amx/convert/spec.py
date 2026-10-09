"""One-line architecture specs for plain state_dicts (same grammar as the C loader):

    "input(64) dense(fc1) relu dense(fc2) relu dense(out) softmax"
    "input(120) conv1d(c1, stride=2, padding=1) relu flatten dense(head)"
    "dense(enc) tanh gru(rnn) dense(pi)"          # PyTorch nn.GRU names: rnn.weight_ih_l0 ...
    "dense(fc1) batchnorm(bn, eps=1e-5) leaky_relu(0.2) dense(fc2)"
"""
from __future__ import annotations

import re

import numpy as np

from ..format import Model, read_safetensors

_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?")


def parse_spec(spec: str) -> tuple[list[dict], int | None]:
    layers, inputs = [], None
    for op, args in _TOKEN.findall(spec):
        parts = [a.strip() for a in args.split(",") if a.strip()] if args else []
        name = parts[0] if parts else ""
        opts = {}
        for p in parts[1:]:
            k, _, v = p.partition("=")
            opts[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
        if op == "input":
            inputs = int(name)
        elif op in ("dense", "linear"):
            layers.append({"op": "dense", "weight": f"{name}.weight", "bias": f"{name}.bias", "bias_optional": True})
        elif op == "conv1d":
            layers.append({"op": "conv1d", "weight": f"{name}.weight", "bias": f"{name}.bias", "bias_optional": True, **opts})
        elif op == "gru":
            layers.append({"op": "gru", "weight_ih": f"{name}.weight_ih_l0", "weight_hh": f"{name}.weight_hh_l0",
                           "bias_ih": f"{name}.bias_ih_l0", "bias_hh": f"{name}.bias_hh_l0", "bias_optional": True})
        elif op == "layernorm":
            layers.append({"op": "layernorm", "weight": f"{name}.weight", "bias": f"{name}.bias", **opts})
        elif op == "batchnorm":
            layers.append({"op": "batchnorm", "weight": f"{name}.weight", "bias": f"{name}.bias",
                           "mean": f"{name}.running_mean", "var": f"{name}.running_var", **opts})
        elif op == "leaky_relu":
            layers.append({"op": "leaky_relu", "alpha": float(name) if name else 0.01})
        else:
            layers.append({"op": op})
    return layers, inputs


def from_spec(tensors: dict[str, np.ndarray], spec: str, labels: list[str] | None = None) -> Model:
    layers, inputs = parse_spec(spec)
    for l in layers:  # drop optional biases that are absent, like the C loader
        for key in ("bias", "bias_ih", "bias_hh"):
            if key in l and l.get("bias_optional") and l[key] not in tensors:
                del l[key]
        l.pop("bias_optional", None)
    used = {v for l in layers for k, v in l.items() if isinstance(v, str) and k != "op"}
    missing = sorted(n for n in used if n not in tensors)
    if missing:
        raise KeyError(f"spec references missing tensors: {missing}")
    return Model(layers, {k: v for k, v in tensors.items() if k in used}, inputs, labels)


def from_safetensors(path, spec: str, labels: list[str] | None = None) -> Model:
    tensors, _ = read_safetensors(path)
    return from_spec(tensors, spec, labels)

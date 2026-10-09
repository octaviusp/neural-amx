"""Shared builder used by every converter: emits graph layers and names tensors uniquely."""
from __future__ import annotations

import numpy as np

from ..format import Model

F = np.float32


class UnsupportedLayer(ValueError):
    pass


class Builder:
    def __init__(self):
        self.layers: list[dict] = []
        self.tensors: dict[str, np.ndarray] = {}
        self._n = 0

    def _name(self, kind: str) -> str:
        self._n += 1
        return f"l{self._n}_{kind}"

    def _put(self, name: str, arr) -> str:
        self.tensors[name] = np.ascontiguousarray(np.asarray(arr, F))
        return name

    def dense(self, weight_out_in, bias=None):
        p = self._name("dense")
        layer = {"op": "dense", "weight": self._put(f"{p}.weight", weight_out_in)}
        if bias is not None:
            layer["bias"] = self._put(f"{p}.bias", bias)
        self.layers.append(layer)

    def conv1d(self, weight_out_in_k, bias=None, stride=1, padding=0):
        p = self._name("conv1d")
        layer = {"op": "conv1d", "weight": self._put(f"{p}.weight", weight_out_in_k), "stride": int(stride), "padding": int(padding)}
        if bias is not None:
            layer["bias"] = self._put(f"{p}.bias", bias)
        self.layers.append(layer)

    def gru(self, w_ih, w_hh, b_ih=None, b_hh=None):
        """PyTorch layout and gate order (r, z, n)."""
        p = self._name("gru")
        layer = {"op": "gru", "weight_ih": self._put(f"{p}.weight_ih", w_ih), "weight_hh": self._put(f"{p}.weight_hh", w_hh)}
        if b_ih is not None:
            layer["bias_ih"] = self._put(f"{p}.bias_ih", b_ih)
        if b_hh is not None:
            layer["bias_hh"] = self._put(f"{p}.bias_hh", b_hh)
        self.layers.append(layer)

    def layernorm(self, gamma=None, beta=None, eps=1e-5):
        p = self._name("layernorm")
        layer = {"op": "layernorm", "eps": float(eps)}
        if gamma is not None:
            layer["weight"] = self._put(f"{p}.weight", gamma)
        if beta is not None:
            layer["bias"] = self._put(f"{p}.bias", beta)
        self.layers.append(layer)

    def batchnorm(self, mean, var, gamma=None, beta=None, eps=1e-5):
        p = self._name("batchnorm")
        layer = {"op": "batchnorm", "mean": self._put(f"{p}.mean", mean), "var": self._put(f"{p}.var", var), "eps": float(eps)}
        if gamma is not None:
            layer["weight"] = self._put(f"{p}.weight", gamma)
        if beta is not None:
            layer["bias"] = self._put(f"{p}.bias", beta)
        self.layers.append(layer)

    def affine(self, scale, shift=None):
        p = self._name("affine")
        layer = {"op": "affine", "scale": self._put(f"{p}.scale", scale)}
        if shift is not None:
            layer["shift"] = self._put(f"{p}.shift", shift)
        self.layers.append(layer)

    def transpose(self, rows: int, cols: int):
        self.layers.append({"op": "transpose", "rows": int(rows), "cols": int(cols)})

    def act(self, op: str, **opts):
        if op in ("identity", "linear", None):
            return
        self.layers.append({"op": op, **opts})

    def model(self, inputs: int | None, labels=None) -> Model:
        if not self.layers:
            raise UnsupportedLayer("the model has no supported layers")
        return Model(self.layers, self.tensors, int(inputs) if inputs else None, list(labels) if labels else None)


def gelu_op(approximate: str | bool | None) -> str:
    if approximate in (None, False, "none", "precise_erf"):
        return "gelu"
    if approximate in (True, "tanh", "precise"):
        return "gelu_tanh"
    raise UnsupportedLayer(f"GELU approximation {approximate!r} is not supported (use 'none' or 'tanh')")


def same_padding(kernel: int, stride: int, dilation: int = 1) -> int:
    if stride != 1 or dilation != 1 or kernel % 2 == 0:
        raise UnsupportedLayer("'same' padding needs stride 1 and an odd kernel")
    return (kernel - 1) // 2

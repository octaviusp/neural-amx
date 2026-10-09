"""Quantization-aware training that matches the runtime: per-row symmetric weights (int8 or int4) and dynamic
per-sample int8 activations, with straight-through gradients. Int4 needs QAT to keep accuracy on small networks;
int8 is usually lossless without it.

PyTorch:
    model = neural_amx.qat.torch_prepare(model, "int4")   # swaps nn.Linear for QAT layers (same weights)
    ... train as usual ...
    neural_amx.from_torch(model)                           # QAT layers convert like nn.Linear
MLX:
    model = neural_amx.qat.mlx_prepare(model, "int4")
"""
from __future__ import annotations


def _bits_plan(n: int, precision: str) -> list[int]:
    if precision == "int8":
        return [8] * n
    if precision == "int4":
        return [8 if i in (0, n - 1) else 4 for i in range(n)]
    raise ValueError("precision must be 'int8' or 'int4'")


# ---------- PyTorch ----------
def torch_fake_quant_weight(w, bits: int):
    import torch
    qmax = float(2 ** (bits - 1) - 1)
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.floor(w / s + 0.5), -qmax, qmax) * s
    return w + (q - w).detach()


def torch_fake_quant_act(x):
    import torch
    s = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.floor(x / s + 0.5), -127.0, 127.0) * s
    return x + (q - x).detach()


def torch_prepare(model, precision: str = "int4"):
    import torch.nn as nn
    import torch.nn.functional as Fn

    class QATLinear(nn.Linear):
        bits = 8

        def forward(self, x):
            return Fn.linear(torch_fake_quant_act(x), torch_fake_quant_weight(self.weight, self.bits), self.bias)

    linears = [m for m in model.modules() if type(m) is nn.Linear]
    for m, bits in zip(linears, _bits_plan(len(linears), precision)):
        m.__class__ = QATLinear  # keeps parameters, optimizer state and module tree intact
        m.bits = bits
    return model


# ---------- MLX ----------
def mlx_fake_quant_weight(w, bits: int):
    import mlx.core as mx
    qmax = float(2 ** (bits - 1) - 1)
    s = mx.maximum(mx.max(mx.abs(mx.stop_gradient(w)), axis=1, keepdims=True), 1e-12) / qmax
    q = mx.clip(mx.floor(w / s + 0.5), -qmax, qmax) * s
    return w + mx.stop_gradient(q - w)


def mlx_fake_quant_act(x):
    import mlx.core as mx
    s = mx.maximum(mx.max(mx.abs(mx.stop_gradient(x)), axis=-1, keepdims=True), 1e-12) / 127.0
    q = mx.clip(mx.floor(x / s + 0.5), -127.0, 127.0) * s
    return x + mx.stop_gradient(q - x)


def mlx_prepare(model, precision: str = "int4"):
    import mlx.nn as nn

    class QATLinear(nn.Linear):
        def __call__(self, x):
            w = mlx_fake_quant_weight(self.weight, self._qat_bits)
            y = mlx_fake_quant_act(x) @ w.T
            return y + self.bias if "bias" in self else y

    def in_order(m):  # execution order for Sequential / lists, declaration order otherwise
        if isinstance(m, nn.Sequential):
            for c in m.layers:
                yield from in_order(c)
        elif isinstance(m, (list, tuple)):
            for c in m:
                yield from in_order(c)
        elif type(m) is nn.Linear:
            yield m
        elif isinstance(m, nn.Module):
            for c in m.children().values():
                yield from in_order(c)

    linears = list(in_order(model))
    for m, bits in zip(linears, _bits_plan(len(linears), precision)):
        m.__class__ = QATLinear
        m._qat_bits = bits
    return model

"""MLX -> neural-amx. Accepts mlx.nn.Sequential (nested) or a list of layers/functions.

MLX Conv1d is channels-last (N, L, C); put `Flatten()` from this module after the conv block (MLX has no flatten
layer) and pass `inputs=length*channels` plus `channels=`; the converter inserts the needed transposes.
"""
from __future__ import annotations

import numpy as np

from .common import Builder, UnsupportedLayer, gelu_op


class Flatten:
    """Marker for mlx.nn.Sequential: (N, L, C) -> (N, L*C), the MLX/NumPy row-major order."""

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)


def _np(a):
    return None if a is None else np.asarray(a, dtype=np.float32)


def _layers(obj):
    import mlx.nn as nn
    if isinstance(obj, (list, tuple)):
        for m in obj:
            yield from _layers(m)
    elif isinstance(obj, nn.Sequential):
        yield from _layers(obj.layers)
    else:
        yield obj


def from_mlx(model, inputs: int | None = None, labels=None, channels: int | None = None):
    import mlx.core as mx
    import mlx.nn as nn
    b = Builder()
    funcs = {nn.relu: "relu", nn.sigmoid: "sigmoid", mx.sigmoid: "sigmoid", nn.tanh: "tanh", mx.tanh: "tanh",
             nn.silu: "silu", nn.gelu: "gelu", nn.softmax: "softmax", mx.softmax: "softmax"}
    layout = None
    first_in = None
    for m in _layers(model):
        p = dict(m.parameters()) if isinstance(m, nn.Module) else {}
        if isinstance(m, nn.Linear):
            first_in = first_in or m.weight.shape[1]
            b.dense(_np(m.weight), _np(p.get("bias")))
        elif isinstance(m, nn.Conv1d):
            if getattr(m, "groups", 1) != 1 or getattr(m, "dilation", 1) not in (1, (1,), [1]):
                raise UnsupportedLayer("Conv1d: only groups=1, dilation=1")
            w = _np(m.weight)  # (out, k, in)
            if layout is None:
                ch = channels or w.shape[2]
                if not inputs or inputs % ch:
                    raise UnsupportedLayer("conv models need inputs=length*channels")
                b.transpose(inputs // ch, ch)
                layout = (inputs // ch, ch)
            stride = m.stride if isinstance(m.stride, int) else m.stride[0]
            pad = m.padding if isinstance(m.padding, int) else m.padding[0]
            b.conv1d(np.transpose(w, (0, 2, 1)), _np(p.get("bias")), stride, pad)
            layout = ((layout[0] + 2 * pad - w.shape[1]) // stride + 1, w.shape[0])
        elif isinstance(m, Flatten):
            if layout:
                b.transpose(layout[1], layout[0])
                layout = None
        elif isinstance(m, nn.GRU):
            first_in = first_in or m.Wx.shape[1]
            H = m.hidden_size
            bhn = p.get("bhn")
            b_hh = None if bhn is None else np.concatenate([np.zeros(2 * H, np.float32), _np(bhn)])
            b.gru(_np(m.Wx), _np(m.Wh), _np(p.get("b")), b_hh)
        elif isinstance(m, nn.LayerNorm):
            b.layernorm(_np(p.get("weight")), _np(p.get("bias")), m.eps)
        elif isinstance(m, nn.BatchNorm):
            b.batchnorm(_np(m.running_mean), _np(m.running_var), _np(p.get("weight")), _np(p.get("bias")), m.eps)
        elif isinstance(m, nn.ReLU):
            b.act("relu")
        elif isinstance(m, nn.LeakyReLU):
            b.act("leaky_relu", alpha=float(m._negative_slope))
        elif isinstance(m, nn.Sigmoid):
            b.act("sigmoid")
        elif isinstance(m, nn.Tanh):
            b.act("tanh")
        elif isinstance(m, nn.GELU):
            approx = getattr(m, "_approx", "none")  # none: erf; precise/tanh: tanh approximation; fast: unsupported
            b.act(gelu_op("tanh" if approx == "precise" else approx))
        elif isinstance(m, nn.SiLU):
            b.act("silu")
        elif isinstance(m, nn.Softmax):
            b.act("softmax")
        elif isinstance(m, (nn.Dropout, nn.Identity)):
            continue
        elif m in funcs:
            b.act(funcs[m])
        else:
            raise UnsupportedLayer(f"unsupported MLX layer {type(m).__name__}")
    return b.model(inputs or first_in, labels)

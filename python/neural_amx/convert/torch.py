"""PyTorch -> neural-amx. Accepts nn.Sequential (nested), a list of modules, or a custom module whose forward is a
straight chain (traced with torch.fx). Recurrent policies: pass the per-step layers, e.g. [gru, nn.ReLU(), head];
the runtime keeps one hidden state per agent slot and runs one time step per call.
"""
from __future__ import annotations

import numpy as np

from .common import Builder, UnsupportedLayer, gelu_op, same_padding


def _np(t):
    return None if t is None else t.detach().cpu().float().numpy()


def _modules(obj):
    import torch.nn as nn
    if isinstance(obj, (list, tuple, nn.ModuleList)):
        for m in obj:
            yield from _modules(m)
    elif isinstance(obj, nn.Sequential):
        for m in obj:
            yield from _modules(m)
    else:
        yield obj


def _emit_module(b: Builder, m):
    import torch.nn as nn
    if isinstance(m, nn.Linear):
        b.dense(_np(m.weight), _np(m.bias))
    elif isinstance(m, nn.Conv1d):
        if m.groups != 1 or m.dilation[0] != 1 or m.padding_mode != "zeros":
            raise UnsupportedLayer("Conv1d: only groups=1, dilation=1, zero padding")
        pad = m.padding if isinstance(m.padding, str) else m.padding[0]
        if pad == "valid":
            pad = 0
        elif pad == "same":
            pad = same_padding(m.kernel_size[0], m.stride[0])
        b.conv1d(_np(m.weight), _np(m.bias), m.stride[0], pad)
    elif isinstance(m, nn.GRU):
        if m.num_layers != 1 or m.bidirectional:
            raise UnsupportedLayer("GRU: only num_layers=1, unidirectional (stack several GRU modules instead)")
        b.gru(_np(m.weight_ih_l0), _np(m.weight_hh_l0), _np(getattr(m, "bias_ih_l0", None)), _np(getattr(m, "bias_hh_l0", None)))
    elif isinstance(m, nn.LayerNorm):
        if len(m.normalized_shape) != 1:
            raise UnsupportedLayer("LayerNorm: only over the last dimension")
        b.layernorm(_np(m.weight), _np(m.bias), m.eps)
    elif isinstance(m, nn.BatchNorm1d):
        b.batchnorm(_np(m.running_mean), _np(m.running_var), _np(m.weight), _np(m.bias), m.eps)
    elif isinstance(m, nn.ReLU):
        b.act("relu")
    elif isinstance(m, nn.LeakyReLU):
        b.act("leaky_relu", alpha=float(m.negative_slope))
    elif isinstance(m, nn.Sigmoid):
        b.act("sigmoid")
    elif isinstance(m, nn.Tanh):
        b.act("tanh")
    elif isinstance(m, nn.GELU):
        b.act(gelu_op(m.approximate))
    elif isinstance(m, nn.SiLU):
        b.act("silu")
    elif isinstance(m, nn.Softmax):
        if m.dim not in (-1, 1, None):
            raise UnsupportedLayer("Softmax: only over the feature dimension")
        b.act("softmax")
    elif isinstance(m, (nn.Flatten, nn.Dropout, nn.Identity)):
        b.act("flatten" if isinstance(m, nn.Flatten) else "identity")
    else:
        raise UnsupportedLayer(f"unsupported PyTorch module {type(m).__name__}")


_FUNCS = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "silu": "silu", "softmax": "softmax",
          "leaky_relu": "leaky_relu", "gelu": "gelu", "flatten": "flatten", "view": "flatten", "reshape": "flatten",
          "dropout": "identity", "contiguous": "identity"}


def _trace(module):
    """Linearizes a custom module with torch.fx: a chain of submodules and elementwise functions."""
    import torch.fx
    gm = torch.fx.symbolic_trace(module)
    steps, prev = [], None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            prev = node
            continue
        if node.op == "output":
            break
        args = [a for a in node.args if isinstance(a, torch.fx.Node)]
        if args and args[0] is not prev:
            raise UnsupportedLayer(f"{module.__class__.__name__}: forward is not a straight chain at '{node.name}'")
        if node.op == "call_module":
            steps.append(("module", gm.get_submodule(node.target)))
        else:
            name = getattr(node.target, "__name__", str(node.target)).lstrip("_")
            if name not in _FUNCS:
                raise UnsupportedLayer(f"unsupported operation '{name}' in forward")
            steps.append(("func", (_FUNCS[name], node.kwargs, node.args)))
        prev = node
    return steps


def from_torch(model, inputs: int | None = None, labels=None):
    import torch.nn as nn
    b = Builder()
    if isinstance(model, (nn.Sequential, list, tuple, nn.ModuleList)) or type(model) in vars(nn).values():
        for m in _modules(model):
            _emit_module(b, m)
    else:
        for kind, item in _trace(model):
            if kind == "module":
                for m in _modules(item):
                    _emit_module(b, m)
                continue
            op, kwargs, args = item
            if op == "leaky_relu":
                slope = kwargs.get("negative_slope", args[1] if len(args) > 1 else 0.01)
                b.act("leaky_relu", alpha=float(slope))
            elif op == "gelu":
                b.act(gelu_op(kwargs.get("approximate", "none")))
            else:
                b.act(op)
    return b.model(inputs or _infer_inputs(model), labels)


def _infer_inputs(model):
    import torch.nn as nn
    for m in model.modules() if hasattr(model, "modules") else _modules(model):
        if isinstance(m, nn.Linear):
            return m.in_features
        if isinstance(m, nn.GRU):
            return m.input_size
        if isinstance(m, nn.Conv1d):
            raise UnsupportedLayer("conv models need inputs=channels*length")
    return None

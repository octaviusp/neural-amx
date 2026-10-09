"""TensorFlow / Keras 3 -> neural-amx. Sequential or straight-chain functional models.

Keras Conv1D is channels-last; the converter inserts the transposes the runtime needs, so the model takes the same
flat input as Keras (length x channels) and Flatten keeps the Keras order.
"""
from __future__ import annotations

import numpy as np

from .common import Builder, UnsupportedLayer, gelu_op, same_padding

_ACT = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "softmax": "softmax", "silu": "silu", "swish": "silu",
        "linear": None, "gelu": "gelu", "leaky_relu": "leaky_relu"}


def _activation(b: Builder, fn):
    name = getattr(fn, "__name__", str(fn))
    if name not in _ACT:
        raise UnsupportedLayer(f"unsupported Keras activation '{name}'")
    if name == "leaky_relu":
        b.act("leaky_relu", alpha=0.2)  # keras.activations.leaky_relu default negative_slope
    else:
        b.act(_ACT[name])


def _reorder_zrh(m, H, axis=0):
    """Keras gate order (z, r, h) -> PyTorch (r, z, n) along `axis`."""
    z, r, h = np.split(m, [H, 2 * H], axis=axis)
    return np.concatenate([r, z, h], axis=axis)


def from_keras(model, labels=None):
    b = Builder()
    shape = tuple(int(d) for d in model.input_shape[1:])
    inputs = int(np.prod(shape))
    if len(shape) == 2 and any(type(l).__name__ == "GRU" for l in model.layers):
        inputs = shape[1]  # recurrent models run one time step (features) per call
    layout = None  # (length, channels) while activations are channels-first inside a conv block
    for layer in model.layers:
        cls = type(layer).__name__
        w = [np.asarray(v) for v in layer.get_weights()]
        if cls in ("InputLayer", "Dropout", "GaussianNoise", "ActivityRegularization"):
            continue
        if cls == "Dense":
            if layout:
                raise UnsupportedLayer("Dense directly after Conv1D (per position) is not supported; add Flatten")
            b.dense(w[0].T, w[1] if layer.use_bias else None)
            _activation(b, layer.activation)
        elif cls == "Conv1D":
            if layer.dilation_rate[0] != 1 or layer.groups != 1 or layer.data_format != "channels_last":
                raise UnsupportedLayer("Conv1D: only dilation 1, groups 1, channels_last")
            if layout is None:
                length = shape[0] if len(shape) == 2 else None
                if length is None:
                    raise UnsupportedLayer("Conv1D must see a (length, channels) input")
                b.transpose(length, shape[1])  # (L, C) -> (C, L)
                layout = (length, shape[1])
            k, stride = layer.kernel_size[0], layer.strides[0]
            pad = 0 if layer.padding == "valid" else same_padding(k, stride)
            b.conv1d(np.transpose(w[0], (2, 1, 0)), w[1] if layer.use_bias else None, stride, pad)
            out_len = (layout[0] + 2 * pad - k) // stride + 1
            layout = (out_len, w[0].shape[2])
            _activation(b, layer.activation)
        elif cls == "Flatten":
            if layout:
                b.transpose(layout[1], layout[0])  # (C, L) -> Keras (L, C) order
                layout = None
        elif cls == "GRU":
            if not layer.reset_after:
                raise UnsupportedLayer("GRU: only reset_after=True (the Keras 2+ default)")
            if layer.activation.__name__ != "tanh" or layer.recurrent_activation.__name__ != "sigmoid":
                raise UnsupportedLayer("GRU: only tanh / sigmoid activations")
            kernel, rec = w[0], w[1]
            H = rec.shape[0]
            bias = w[2] if layer.use_bias else np.zeros((2, 3 * H), np.float32)
            b.gru(_reorder_zrh(kernel.T, H), _reorder_zrh(rec.T, H), _reorder_zrh(bias[0], H), _reorder_zrh(bias[1], H))
        elif cls == "LayerNormalization":
            axis = layer.axis if isinstance(layer.axis, int) else layer.axis[0]
            if layout or axis not in (-1, len(layer.input.shape) - 1):
                raise UnsupportedLayer("LayerNormalization: only over the last axis of flat features")
            i = 0
            gamma = w[i] if layer.scale else None
            i += int(layer.scale)
            beta = w[i] if layer.center else None
            b.layernorm(gamma, beta, layer.epsilon)
        elif cls == "BatchNormalization":
            i = 0
            gamma = w[i] if layer.scale else None
            i += int(layer.scale)
            beta = w[i] if layer.center else None
            i += int(layer.center)
            mean, var = w[i], w[i + 1]
            rep = (lambda v: None if v is None else np.repeat(v, layout[0])) if layout else (lambda v: v)
            b.batchnorm(rep(mean), rep(var), rep(gamma), rep(beta), layer.epsilon)
        elif cls == "Normalization":
            mean, var = np.asarray(layer.mean).reshape(-1), np.asarray(layer.variance).reshape(-1)
            if mean.size == 1:
                mean, var = np.full(inputs, mean[0]), np.full(inputs, var[0])
            scale = (1.0 / np.sqrt(np.maximum(var, 1e-7))).astype(np.float32)
            b.affine(scale, (-mean * scale).astype(np.float32))
        elif cls == "Activation":
            _activation(b, layer.activation)
        elif cls == "ReLU":
            if layer.max_value is not None or layer.threshold:
                raise UnsupportedLayer("ReLU: max_value/threshold not supported")
            b.act("leaky_relu", alpha=float(layer.negative_slope)) if layer.negative_slope else b.act("relu")
        elif cls == "LeakyReLU":
            b.act("leaky_relu", alpha=float(getattr(layer, "negative_slope", getattr(layer, "alpha", 0.3))))
        elif cls == "Softmax":
            b.act("softmax")
        else:
            raise UnsupportedLayer(f"unsupported Keras layer {cls}")
    return b.model(inputs, labels)


__all__ = ["from_keras", "gelu_op"]

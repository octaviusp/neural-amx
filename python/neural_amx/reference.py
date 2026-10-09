"""Executable specification of the neural-amx runtime (core/*.c), in numpy.

Every float32 operation happens in the same order as the C code, so the C library (any backend: wasm SIMD128, SSE2,
NEON, scalar) must return exactly the same bits as `Runner.run`. Tests enforce it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .format import Model

F = np.float32
PRECISIONS = {"stored": 0, "f32": 32, "int8": 8, "int4": 4}


def _f(v) -> np.float32:
    return np.float32(v)


# ---------- deterministic math (core/nam_math.h) ----------
def expf(x):
    x = np.asarray(x, F)
    x = np.where(x > _f(88.0), _f(88.0), x)
    under = x < _f(-87.0)
    n = np.floor(x * _f(1.44269504088896341) + _f(0.5))
    x = x - n * _f(0.693359375)
    x = x - n * _f(-2.12194440e-4)
    z = x * x
    p = np.full_like(x, _f(1.9875691500e-4))
    for c in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1):
        p = p * x + _f(c)
    p = p * z + x + _f(1.0)
    ni = np.where(np.isfinite(n), n, 0).astype(np.int64)
    pow2 = ((np.clip(ni, -126, 127) + 127).astype(np.uint32) << 23).view(F)
    return np.where(under, _f(0.0), p * pow2).astype(F)


def sigmoidf(x):
    return (_f(1.0) / (_f(1.0) + expf(-np.asarray(x, F)))).astype(F)


def tanhf(x):
    x = np.asarray(x, F)
    a = np.abs(x)
    x2 = x * x
    small = x + x * x2 * (_f(-0.333333333) + x2 * _f(0.133333333))
    e = expf(_f(2.0) * x)
    mid = (e - _f(1.0)) / (e + _f(1.0))
    big = np.where(x < 0, _f(-1.0), _f(1.0))
    return np.where(a > _f(9.0), big, np.where(a < _f(0.0625), small, mid)).astype(F)


def erff(x):
    x = np.asarray(x, F)
    a = np.abs(x)
    t = _f(1.0) / (_f(1.0) + _f(0.3275911) * a)
    y = np.full_like(x, _f(1.061405429))
    for c in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
        y = y * t + _f(c)
    y = _f(1.0) - y * t * expf(-a * a)
    return np.where(x < 0, -y, y).astype(F)


def gelu_erf(x):
    x = np.asarray(x, F)
    return (_f(0.5) * x * (_f(1.0) + erff(x * _f(0.70710678118654752)))).astype(F)


def gelu_tanh(x):
    x = np.asarray(x, F)
    return (_f(0.5) * x * (_f(1.0) + tanhf(_f(0.79788456080286536) * (x + _f(0.044715) * x * x * x)))).astype(F)


# ---------- weights (core/nam_model.c load_matrix) ----------
@dataclass
class Matrix:
    rows: int
    cols: int
    cols_pad: int
    bits: int
    w: np.ndarray | None  # float32 [rows, cols_pad]
    q: np.ndarray | None  # int64 [rows, cols_pad]
    scale: np.ndarray | None
    bias: np.ndarray


def pad16(n: int) -> int:
    return (n + 15) & ~15


def unpack_int4(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    p = packed.reshape(rows, -1).astype(np.int64)
    out = np.empty((rows, p.shape[1] * 2), np.int64)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = p >> 4
    out = np.where(out >= 8, out - 16, out)
    return out[:, :cols]


def quantize_weights(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric: s = maxabs / qmax, q = clamp(floor(w / s + 0.5)). Same as the C loader."""
    qmax = _f(127.0 if bits == 8 else 7.0)
    w = np.asarray(w, F)
    s = (np.abs(w).max(axis=1) / qmax).astype(F)
    safe = np.where(s > 0, s, _f(1.0))
    q = np.where(s[:, None] > 0, np.floor(w / safe[:, None] + _f(0.5)), _f(0.0))
    return np.clip(q, -qmax, qmax).astype(np.int64), s


def load_matrix(tensors: dict, name: str, rows: int, cols: int, target_bits: int) -> Matrix:
    if name not in tensors:
        raise KeyError(f"tensor '{name}' not found")
    t = tensors[name]
    q = scale = None
    stored = 32
    if t.dtype.kind == "f":
        if t.size != rows * cols:
            raise ValueError(f"'{name}' has {t.size} values, expected {rows}x{cols}")
        w = t.astype(F).reshape(rows, cols)
    elif t.dtype in (np.int8, np.uint8):
        scale = tensors[name + ".scale"].astype(F).reshape(rows)
        stored = 8 if t.dtype == np.int8 else 4
        q = t.astype(np.int64).reshape(rows, cols) if stored == 8 else unpack_int4(t, rows, cols)
        w = (q.astype(F) * scale[:, None]).astype(F)
    else:
        raise TypeError(f"'{name}': unsupported dtype {t.dtype}")
    bits = target_bits or stored
    if bits != 32 and not (q is not None and stored <= bits):
        q, scale = quantize_weights(w, bits)
    cp = pad16(cols)
    if bits == 32:
        wp = np.zeros((rows, cp), F)
        wp[:, :cols] = w
        return Matrix(rows, cols, cp, 32, wp, None, None, np.zeros(rows, F))
    qp = np.zeros((rows, cp), np.int64)
    qp[:, :cols] = q
    return Matrix(rows, cols, cp, bits, None, qp, scale.astype(F), np.zeros(rows, F))


# ---------- kernels (core/nam_kernels.c) ----------
def dense_float(m: Matrix, x: np.ndarray) -> np.ndarray:
    """x: (n, >= cols_pad) float32, zero padded. Lanes a/b over blocks of 8, then (t0 + t1) + (t2 + t3)."""
    n = x.shape[0]
    a = np.zeros((n, m.rows, 4), F)
    b = np.zeros((n, m.rows, 4), F)
    for k in range(0, m.cols_pad, 8):
        a = a + m.w[None, :, k:k + 4] * x[:, None, k:k + 4]
        b = b + m.w[None, :, k + 4:k + 8] * x[:, None, k + 4:k + 8]
    t = a + b
    return ((t[..., 0] + t[..., 1]) + (t[..., 2] + t[..., 3]) + m.bias[None, :]).astype(F)


def quantize_rows(x: np.ndarray, n: int, npad: int) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic per-row int8 activations (nam_quantize_row)."""
    v = np.asarray(x[:, :n], F)
    a = np.abs(v)
    a = np.where(np.isnan(a), _f(0.0), a)
    maxabs = a.max(axis=1) if n else np.zeros(len(v), F)
    bad = ~(maxabs > 0) | (maxabs > _f(3.4e38))
    inv = _f(127.0) / np.where(bad, _f(1.0), maxabs)
    t = np.floor(v * inv[:, None] + _f(0.5))
    t = np.clip(np.where(np.isnan(t), _f(0.0), t), -127, 127)
    q = np.zeros((len(v), npad), np.int64)
    q[:, :n] = t.astype(np.int64)
    q[bad] = 0
    return q, np.where(bad, _f(0.0), maxabs / _f(127.0)).astype(F)


def dense_int(m: Matrix, q: np.ndarray, sx: np.ndarray) -> np.ndarray:
    acc = (q[:, :m.cols_pad] @ m.q.T).astype(np.int64)
    if np.abs(acc).max(initial=0) >= 2 ** 31:
        raise OverflowError("int32 accumulator overflow")
    return (acc.astype(F) * (sx[:, None] * m.scale[None, :]) + m.bias[None, :]).astype(F)


def dense(m: Matrix, x: np.ndarray) -> np.ndarray:
    if m.bits == 32:
        xp = np.zeros((len(x), m.cols_pad), F)
        xp[:, :m.cols] = x[:, :m.cols]
        return dense_float(m, xp)
    q, sx = quantize_rows(x, m.cols, m.cols_pad)
    return dense_int(m, q, sx)


# ---------- runner (core/nam_run.c) ----------
def _seq_sum(v: np.ndarray) -> np.float32:
    return np.cumsum(v.astype(F), dtype=F)[-1] if len(v) else _f(0.0)


class Runner:
    def __init__(self, model: Model, precision: str = "stored", slots: int = 33):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)}")
        self.layers, self.slots = [], slots
        tensors = model.tensors
        ops = [l for l in model.layers if l["op"] not in ("flatten", "identity", "dropout")]
        weighted = sum(l["op"] in ("dense", "linear", "conv1d", "gru") for l in ops)
        n_in = model.inputs or self._infer_inputs(ops, tensors)
        self.inputs, k = n_in, 0
        for i, spec in enumerate(ops):
            op = "dense" if spec["op"] == "linear" else spec["op"]
            bits = 0
            if op in ("dense", "conv1d", "gru"):
                bits = self._target_bits(precision, k, weighted, op == "gru")
                if precision == "stored" and spec.get("bits"):
                    bits = int(spec["bits"])
                k += 1
            layer = {"op": op, "in": n_in, "out": n_in}
            try:
                self._build(layer, spec, tensors, n_in, bits)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"layer {i}: {e}") from e
            self.layers.append(layer)
            n_in = layer["out"]
        self.outputs = n_in
        self.state = {i: np.zeros((slots, l["hidden"]), F) for i, l in enumerate(self.layers) if l["op"] == "gru"}

    @staticmethod
    def _infer_inputs(ops, tensors):
        if ops and ops[0]["op"] in ("dense", "linear", "gru"):
            return int(tensors[ops[0]["weight_ih" if ops[0]["op"] == "gru" else "weight"]].shape[1])
        raise ValueError("cannot infer the input size; set Model.inputs")

    @staticmethod
    def _target_bits(precision, k, count, recurrent):
        if precision == "f32":
            return 32
        if precision == "int8":
            return 8
        if precision == "int4":
            return 8 if (recurrent or k == 0 or k == count - 1) else 4
        return 0

    @staticmethod
    def _vec(tensors, name, n, fill, optional=True):
        if name is None or name not in tensors:
            if not optional and name is not None:
                raise KeyError(f"tensor '{name}' not found")
            return np.full(n, _f(fill), F)
        v = tensors[name].astype(F).reshape(-1)
        if v.size != n:
            raise ValueError(f"'{name}' must have {n} values")
        return v

    @staticmethod
    def _bias(m, tensors, spec, key):
        name = spec.get(key)
        if name is None:
            return
        if name not in tensors:
            if spec.get("bias_optional"):
                return
            raise KeyError(f"tensor '{name}' not found")
        m.bias = tensors[name].astype(F).reshape(m.rows)

    def _build(self, L, spec, tensors, n_in, bits):
        op = L["op"]
        if op == "dense":
            rows = int(tensors[spec["weight"]].shape[0])
            L["m"] = load_matrix(tensors, spec["weight"], rows, n_in, bits)
            self._bias(L["m"], tensors, spec, "bias")
            L["out"] = rows
        elif op == "conv1d":
            t = tensors[spec["weight"]]
            out_ch = int(t.shape[0])
            in_ch = int(spec.get("in_channels", t.shape[1] if t.ndim == 3 else 0))
            k = int(spec.get("kernel_size", t.shape[2] if t.ndim == 3 else 0))
            stride, pad = int(spec.get("stride", 1)), int(spec.get("padding", 0))
            if in_ch <= 0 or k <= 0 or n_in % in_ch:
                raise ValueError("conv1d expects input channels x length")
            len_in = n_in // in_ch
            len_out = (len_in + 2 * pad - k) // stride + 1
            L.update(in_ch=in_ch, out_ch=out_ch, kernel=k, stride=stride, pad=pad, len_in=len_in, len_out=len_out)
            L["m"] = load_matrix(tensors, spec["weight"], out_ch, in_ch * k, bits)
            self._bias(L["m"], tensors, spec, "bias")
            L["out"] = out_ch * len_out
        elif op == "gru":
            H = int(tensors[spec["weight_hh"]].shape[0]) // 3
            L["m"] = load_matrix(tensors, spec["weight_ih"], 3 * H, n_in, bits)
            L["h"] = load_matrix(tensors, spec["weight_hh"], 3 * H, H, bits)
            self._bias(L["m"], tensors, spec, "bias_ih")
            self._bias(L["h"], tensors, spec, "bias_hh")
            L["hidden"] = H
            L["out"] = H
        elif op == "layernorm":
            L["eps"] = _f(spec.get("eps", 1e-5))
            L["g"] = self._vec(tensors, spec.get("weight"), n_in, 1.0)
            L["b"] = self._vec(tensors, spec.get("bias"), n_in, 0.0)
        elif op == "batchnorm":
            eps = _f(spec.get("eps", 1e-5))
            w = self._vec(tensors, spec.get("weight"), n_in, 1.0)
            b = self._vec(tensors, spec.get("bias"), n_in, 0.0)
            mean = self._vec(tensors, spec["mean"], n_in, 0.0, optional=False)
            var = self._vec(tensors, spec["var"], n_in, 1.0, optional=False)
            s = (w / np.sqrt(var + eps)).astype(F)
            L.update(op="affine", g=s, b=(b - mean * s).astype(F))
        elif op == "affine":
            L["g"] = self._vec(tensors, spec["scale"], n_in, 1.0, optional=False)
            L["b"] = self._vec(tensors, spec.get("shift"), n_in, 0.0)
        elif op == "leaky_relu":
            L["alpha"] = _f(spec.get("alpha", 0.01))
        elif op not in ("relu", "sigmoid", "tanh", "gelu", "gelu_tanh", "silu", "softmax"):
            raise ValueError(f"unknown op '{op}'")

    def reset(self, slot: int = -1):
        for s in self.state.values():
            if slot < 0:
                s[:] = 0
            else:
                s[slot] = 0

    def run(self, x, first_slot: int = 0) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, F))
        if x.shape[1] != self.inputs:
            raise ValueError(f"expected {self.inputs} inputs, got {x.shape[1]}")
        n = len(x)
        for i, L in enumerate(self.layers):
            op = L["op"]
            if op == "dense":
                x = dense(L["m"], x)
            elif op == "conv1d":
                x = np.stack([self._conv1d(L, row) for row in x])
            elif op == "gru":
                st = self.state[i]
                out = np.empty((n, L["hidden"]), F)
                for r in range(n):
                    out[r] = self._gru(L, x[r:r + 1], st, first_slot + r)
                x = out
            else:
                x = self._elementwise(L, x)
        return x.astype(F)

    def _conv1d(self, L, row):
        m = L["m"]
        cols = L["in_ch"] * L["kernel"]
        idx = np.arange(L["len_out"])[:, None] * L["stride"] + np.arange(L["kernel"])[None, :] - L["pad"]
        inside = (idx >= 0) & (idx < L["len_in"])
        cidx = np.clip(idx, 0, L["len_in"] - 1)
        if m.bits == 32:
            src = row.reshape(L["in_ch"], L["len_in"])
            patch = np.where(inside[:, None, :], src[:, cidx].transpose(1, 0, 2), _f(0.0)).reshape(L["len_out"], cols)
            xp = np.zeros((L["len_out"], m.cols_pad), F)
            xp[:, :cols] = patch
            y = dense_float(m, xp)
        else:
            q, sx = quantize_rows(row[None, :], L["in"], L["in"])
            src = q[0].reshape(L["in_ch"], L["len_in"])
            patch = np.where(inside[:, None, :], src[:, cidx].transpose(1, 0, 2), 0).reshape(L["len_out"], cols)
            qp = np.zeros((L["len_out"], m.cols_pad), np.int64)
            qp[:, :cols] = patch
            y = dense_int(m, qp, np.full(L["len_out"], sx[0], F))
        return y.T.reshape(-1)

    @staticmethod
    def _gru(L, x, st, slot):
        H = L["hidden"]
        h = st[slot:slot + 1]
        gi = dense(L["m"], x)[0]
        gh = dense(L["h"], h)[0]
        r = sigmoidf(gi[:H] + gh[:H])
        z = sigmoidf(gi[H:2 * H] + gh[H:2 * H])
        n = tanhf(gi[2 * H:] + r * gh[2 * H:])
        v = ((_f(1.0) - z) * n + z * h[0]).astype(F)
        st[slot] = v
        return v

    @staticmethod
    def _elementwise(L, x):
        op = L["op"]
        if op == "relu":
            return np.where(x > 0, x, _f(0.0)).astype(F)
        if op == "leaky_relu":
            return np.where(x < 0, x * L["alpha"], x).astype(F)
        if op == "sigmoid":
            return sigmoidf(x)
        if op == "tanh":
            return tanhf(x)
        if op == "gelu":
            return gelu_erf(x)
        if op == "gelu_tanh":
            return gelu_tanh(x)
        if op == "silu":
            return (x * sigmoidf(x)).astype(F)
        if op == "affine":
            return (x * L["g"][None, :] + L["b"][None, :]).astype(F)
        if op == "layernorm":
            out = np.empty_like(x)
            n = _f(x.shape[1])
            for r, row in enumerate(x):
                mean = _seq_sum(row) / n
                d = row - mean
                inv = _f(1.0) / np.sqrt(_seq_sum(d * d) / n + L["eps"])
                out[r] = (d * inv) * L["g"] + L["b"]
            return out
        if op == "softmax":
            out = np.empty_like(x)
            for r, row in enumerate(x):
                mx = row[0]
                for v in row[1:]:
                    if v > mx:
                        mx = v
                e = expf(row - mx)
                out[r] = e / _seq_sum(e)
            return out
        raise ValueError(op)


def run(model: Model, x, precision: str = "stored") -> np.ndarray:
    """One-shot reference inference (fresh recurrent state)."""
    return Runner(model, precision).run(x)

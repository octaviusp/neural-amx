"""ONNX -> neural-amx (straight-chain graphs). Covers models exported from PyTorch (torch.onnx.export),
TensorFlow (tf2onnx), scikit-learn (skl2onnx) and others.

Supported nodes: Gemm, MatMul(+Add), Conv (1D), GRU (linear_before_reset=1), LayerNormalization,
BatchNormalization, Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Softmax, Flatten/Reshape/Squeeze/Unsqueeze/Identity/
Dropout/Cast (pass-through), and decomposed activations (GELU, SiLU, ...) recognized numerically.
"""
from __future__ import annotations

import numpy as np

from .common import Builder, UnsupportedLayer, gelu_op, same_padding

_PASS = {"Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity", "Dropout", "Cast"}
# inputs the runtime does not need: initial/sequence state of GRU, shape and axes operands
_IGNORED_INPUTS = {"GRU": {4, 5}, "Reshape": {1}, "Squeeze": {1}, "Unsqueeze": {1}, "Expand": {1}}


def _live_nodes(g, nodes):
    """Nodes on the path to the graph output, skipping side branches that only feed ignored inputs."""
    producer = {o: n for n in nodes for o in n.output}
    live, stack = set(), [o.name for o in g.output]
    while stack:
        n = producer.get(stack.pop())
        if n is None or id(n) in live:
            continue
        live.add(id(n))
        skip = _IGNORED_INPUTS.get(n.op_type, set())
        stack += [x for k, x in enumerate(n.input) if x and k not in skip]
    return [n for n in nodes if id(n) in live]
_ACT = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Softmax": "softmax"}


def _attrs(node):
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def from_onnx(path_or_model, labels=None):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(path_or_model) if isinstance(path_or_model, (str, bytes)) or hasattr(path_or_model, "__fspath__") else path_or_model
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(_attrs(n)["value"]).astype(np.float32)
    data_inputs = [i for i in g.input if i.name not in consts]
    if len(data_inputs) != 1:
        raise UnsupportedLayer("ONNX model must have exactly one data input")
    dims = [d.dim_value for d in data_inputs[0].type.tensor_type.shape.dim]
    inputs = int(np.prod([d for d in dims[1:] if d > 0]))
    nodes = _live_nodes(g, [n for n in g.node if n.op_type != "Constant"])
    if len(dims) == 3 and any(n.op_type == "GRU" for n in nodes):
        inputs = dims[-1]  # recurrent models run one time step (features) per call
    b, cur, i = Builder(), data_inputs[0].name, 0
    pending_bias = None  # MatMul waiting for its Add

    def data_arg(n):
        skip = _IGNORED_INPUTS.get(n.op_type, set())
        names = [x for k, x in enumerate(n.input) if x and x not in consts and k not in skip]
        if names != [cur]:
            raise UnsupportedLayer(f"node {n.name or n.op_type}: graph is not a straight chain")
        return [consts.get(x) for x in n.input]

    def flush_matmul():
        nonlocal pending_bias
        if pending_bias is not None:
            b.dense(pending_bias, None)
            pending_bias = None

    while i < len(nodes):
        n = nodes[i]
        op = n.op_type
        if op == "Add" and pending_bias is not None:
            args = data_arg(n)
            bias = next(a for a in args if a is not None)
            b.dense(pending_bias, bias.reshape(-1))
            pending_bias = None
            cur = n.output[0]
            i += 1
            continue
        flush_matmul()
        act = _match_activation(nodes, i, cur, consts)
        if act:
            b.act(act[0])
            cur, i = act[1], act[2]
            continue
        args = data_arg(n)
        a = _attrs(n)
        if op == "Gemm":
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                raise UnsupportedLayer("Gemm: only transA=0, alpha=beta=1")
            w = args[1] if a.get("transB", 0) else args[1].T
            b.dense(w, args[2].reshape(-1) if len(args) > 2 and args[2] is not None else None)
        elif op == "MatMul":
            pending_bias = args[1].T  # (in, out) -> (out, in); bias may follow
        elif op == "Conv":
            ks = a.get("kernel_shape", list(args[1].shape[2:]))
            if len(ks) != 1 or a.get("group", 1) != 1 or a.get("dilations", [1])[0] != 1:
                raise UnsupportedLayer("Conv: only 1D, group 1, dilation 1")
            stride = a.get("strides", [1])[0]
            auto = a.get("auto_pad", b"NOTSET")
            auto = auto.decode() if isinstance(auto, bytes) else auto
            if auto in ("SAME_UPPER", "SAME_LOWER"):
                pad = same_padding(ks[0], stride)
            elif auto == "VALID":
                pad = 0
            else:
                pads = a.get("pads", [0, 0])
                if pads[0] != pads[-1]:
                    raise UnsupportedLayer("Conv: only symmetric padding")
                pad = pads[0]
            b.conv1d(args[1], args[2] if len(args) > 2 else None, stride, pad)
        elif op == "GRU":
            if a.get("linear_before_reset", 0) != 1 or a.get("direction", "forward") != "forward":
                raise UnsupportedLayer("GRU: only linear_before_reset=1, forward")
            W, R = args[1][0], args[2][0]
            H = R.shape[1]
            B = args[3][0] if len(args) > 3 and args[3] is not None else np.zeros(6 * H, np.float32)
            zrh = lambda m: np.concatenate([m[H:2 * H], m[:H], m[2 * H:]])  # (z, r, h) -> (r, z, n)
            b.gru(zrh(W), zrh(R), zrh(B[:3 * H]), zrh(B[3 * H:]))
        elif op == "LayerNormalization":
            if a.get("axis", -1) not in (-1, 1):
                raise UnsupportedLayer("LayerNormalization: only the last axis")
            b.layernorm(args[1], args[2] if len(args) > 2 else None, a.get("epsilon", 1e-5))
        elif op == "BatchNormalization":
            b.batchnorm(args[3], args[4], args[1], args[2], a.get("epsilon", 1e-5))
        elif op in _ACT:
            if op == "Softmax" and a.get("axis", -1) not in (-1, 1):
                raise UnsupportedLayer("Softmax: only the feature axis")
            b.act(_ACT[op])
        elif op == "LeakyRelu":
            b.act("leaky_relu", alpha=float(a.get("alpha", 0.01)))
        elif op == "Gelu":
            b.act(gelu_op(a.get("approximate", "none")))
        elif op == "Transpose":
            perm = list(a.get("perm", []))
            if not perm or perm[-1] != len(perm) - 1:  # only batch/time axes may move; features stay last
                raise UnsupportedLayer("Transpose: only permutations that keep the feature axis last")
        elif op not in _PASS:
            raise UnsupportedLayer(f"unsupported ONNX op {op}")
        cur = n.output[0]
        if op == "GRU" and len(n.output) > 1 and n.output[1]:  # Y and Y_h hold the same step; follow the one used
            later = {x for m in nodes[i + 1:] for x in m.input}
            if n.output[0] not in later and n.output[1] in later:
                cur = n.output[1]
        i += 1
    flush_matmul()
    return b.model(inputs, labels)


_ELEMENTWISE = {"Mul", "Add", "Sub", "Div", "Pow", "Erf", "Tanh", "Sigmoid", "Sqrt", "Exp", "Neg", "Reciprocal"}


def _eval_block(block, cur, consts, x):
    env = {cur: x}
    ops = {"Mul": np.multiply, "Add": np.add, "Sub": np.subtract, "Div": np.divide, "Pow": np.power}
    for n in block:
        a = [env[i] if i in env else consts[i] for i in n.input]
        t = n.op_type
        if t in ops:
            env[n.output[0]] = ops[t](a[0], a[1])
        elif t == "Erf":
            from math import erf
            env[n.output[0]] = np.vectorize(erf)(a[0])
        elif t == "Tanh":
            env[n.output[0]] = np.tanh(a[0])
        elif t == "Sigmoid":
            env[n.output[0]] = 1 / (1 + np.exp(-a[0]))
        elif t == "Sqrt":
            env[n.output[0]] = np.sqrt(a[0])
        elif t == "Exp":
            env[n.output[0]] = np.exp(a[0])
        elif t == "Neg":
            env[n.output[0]] = -a[0]
        else:
            env[n.output[0]] = 1 / a[0]
    return env[block[-1].output[0]]


def _match_activation(nodes, i, cur, consts):
    """Collapses a decomposed activation (e.g. exported GELU or SiLU): the longest run of elementwise nodes fed only
    by `cur` and constants whose single live output is identified numerically. Returns (op, output, next index)."""
    from math import erf
    block, produced = [], {cur}
    for n in nodes[i:]:
        if n.op_type not in _ELEMENTWISE or not all(x in produced or x in consts for x in n.input if x):
            break
        block.append(n)
        produced.add(n.output[0])
    later = {x for n in nodes[i + len(block):] for x in n.input}
    while len(block) >= 2:
        outs = {n.output[0] for n in block}
        if outs & later <= {block[-1].output[0]}:
            x = np.linspace(-6, 6, 241)
            y = _eval_block(block, cur, consts, x)
            erf_v = np.vectorize(erf)
            candidates = {"gelu": 0.5 * x * (1 + erf_v(x / np.sqrt(2))),
                          "gelu_tanh": 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))),
                          "silu": x / (1 + np.exp(-x)), "sigmoid": 1 / (1 + np.exp(-x)), "tanh": np.tanh(x)}
            for op, ref in candidates.items():
                if np.allclose(y, ref, atol=1e-6, rtol=1e-5):
                    return op, block[-1].output[0], i + len(block)
        block.pop()
    return None

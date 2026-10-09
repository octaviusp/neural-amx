"""Bit-exactness test: every C build must return exactly the reference's bits for every model, precision, kernel and
batch choice.   uv run --with numpy python -I tests/golden.py <work dir> <runner command>...
A runner command is a shell-split string, e.g. "build/runner-neon" or "node build/runner-wasm-simd.js".
"""
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import neural_amx as nam  # noqa: E402
from neural_amx.format import Model  # noqa: E402

RNG = np.random.default_rng(1234)
F = np.float32


def rand(*shape, scale=0.5):
    return (RNG.standard_normal(shape) * scale).astype(F)


def dense(tensors, name, i, o, bias=True):
    tensors[f"{name}.weight"] = rand(o, i, scale=1 / np.sqrt(i))
    if bias:
        tensors[f"{name}.bias"] = rand(o, scale=0.1)
    return {"op": "dense", "weight": f"{name}.weight", **({"bias": f"{name}.bias"} if bias else {})}


def zoo():
    models = {}
    t = {"norm.scale": rand(64, scale=1) + 1, "norm.shift": rand(64)}
    layers = [{"op": "affine", "scale": "norm.scale", "shift": "norm.shift"}, dense(t, "fc1", 64, 128), {"op": "relu"},
              dense(t, "fc2", 128, 128), {"op": "gelu"}]
    t["ln.weight"], t["ln.bias"] = rand(128, scale=0.2) + 1, rand(128, scale=0.1)
    layers += [{"op": "layernorm", "weight": "ln.weight", "bias": "ln.bias", "eps": 1e-5}, dense(t, "out", 128, 8), {"op": "softmax"}]
    models["mlp"] = (Model(layers, t, 64, [f"a{i}" for i in range(8)]), 1)

    t = {}
    layers = [dense(t, "a", 37, 50), {"op": "leaky_relu", "alpha": 0.1}, dense(t, "b", 50, 33, bias=False), {"op": "sigmoid"},
              dense(t, "c", 33, 21), {"op": "tanh"}, dense(t, "d", 21, 19), {"op": "silu"}, dense(t, "e", 19, 19),
              {"op": "gelu_tanh"}, dense(t, "f", 19, 5)]
    models["acts"] = (Model(layers, t, 37), 1)

    t = {"c1.weight": rand(8, 4, 5, scale=0.3), "c1.bias": rand(8, scale=0.1), "c2.weight": rand(6, 8, 3, scale=0.3)}
    out_len = ((30 + 4 - 5) // 2 + 1) - 3 + 1
    layers = [{"op": "conv1d", "weight": "c1.weight", "bias": "c1.bias", "stride": 2, "padding": 2}, {"op": "relu"},
              {"op": "conv1d", "weight": "c2.weight"}, {"op": "flatten"}, dense(t, "head", 6 * out_len, 3)]
    models["conv"] = (Model(layers, t, 4 * 30), 1)

    t = {"rnn.weight_ih_l0": rand(72, 32, scale=0.3), "rnn.weight_hh_l0": rand(72, 24, scale=0.3),
         "rnn.bias_ih_l0": rand(72, scale=0.1), "rnn.bias_hh_l0": rand(72, scale=0.1)}
    layers = [dense(t, "enc", 20, 32), {"op": "tanh"},
              {"op": "gru", "weight_ih": "rnn.weight_ih_l0", "weight_hh": "rnn.weight_hh_l0", "bias_ih": "rnn.bias_ih_l0", "bias_hh": "rnn.bias_hh_l0"},
              dense(t, "pi", 24, 6)]
    models["gru"] = (Model(layers, t, 20), 5)  # 5 agents per time step

    big = {}
    layers = [dense(big, "l1", 256, 512), {"op": "relu"}, dense(big, "l2", 512, 512), {"op": "relu"}, dense(big, "l3", 512, 32)]
    models["wide"] = (Model(layers, big, 256), 1)
    return models


def inputs_for(name, model, agents):
    n = 13 if agents == 1 else agents * 6
    x = rand(n, model.inputs, scale=1.5)
    x[0] = 0  # zero row: quantizes to zero
    x[1] *= 40  # large values
    return x


def write_inputs(path, x):
    path.write_bytes(np.array(x.shape, np.uint32).tobytes() + x.astype(F).tobytes())


def reference(model, x, precision, agents):
    r = nam.Runner(model, precision)
    return np.concatenate([r.run(x[i:i + agents]) for i in range(0, len(x), agents)])


def main():
    work = Path(sys.argv[1])
    runners = [shlex.split(c) for c in sys.argv[2:]]
    work.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, (model, agents) in zoo().items():
        x = inputs_for(name, model, agents)
        write_inputs(work / f"{name}.inputs.bin", x)
        model.save(work / f"{name}.safetensors")
        for qname in ("int8", "int4"):
            nam.quantize(model, qname).save(work / f"{name}-{qname}q.safetensors")
            cases.append((f"{name}-{qname}q", "stored", agents, x, nam.quantize(model, qname), name))
        for prec in ("stored", "int8", "int4"):
            cases.append((name, prec, agents, x, model, name))
    # a plain state_dict + spec (batchnorm folded at load)
    sd = {"fc1.weight": rand(24, 16, scale=0.25), "fc1.bias": rand(24), "bn.weight": rand(24) + 1, "bn.bias": rand(24),
          "bn.running_mean": rand(24), "bn.running_var": np.abs(rand(24)) + 0.5, "fc2.weight": rand(4, 24, scale=0.2)}
    nam.write_safetensors(work / "plain.safetensors", sd)
    spec = "input(16) dense(fc1) batchnorm(bn) relu dense(fc2) softmax"
    plain = nam.from_spec(sd, spec)
    xp = rand(9, 16, scale=1.0)
    write_inputs(work / "plain.inputs.bin", xp)

    precisions = {"stored": 0, "f32": 1, "int8": 2, "int4": 3}
    failures, checks = [], 0
    jobs = [(c, None) for c in cases] + [(("plain", p, 1, xp, plain, "plain"), spec) for p in ("stored", "int4")]
    for (fname, prec, agents, x, model, inputs_name), spec_arg in jobs:
        expected = reference(model, x, prec, agents)
        if not np.all(np.isfinite(expected)):
            failures.append(f"{fname}/{prec}: reference produced non-finite values")
        for runner in runners:
            for kernel in (1, 2):
                for batch in (0, 1):
                    out = work / "out.bin"
                    cmd = runner + [str(work / f"{fname}.safetensors"), str(work / f"{inputs_name}.inputs.bin"), str(out),
                                    str(precisions[prec]), str(kernel), str(batch), str(agents)] + ([spec_arg] if spec_arg else [])
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    checks += 1
                    tag = f"{' '.join(runner)} {fname}/{prec} kernel={kernel} batch={batch}"
                    if res.returncode:
                        failures.append(f"{tag}: exit {res.returncode} {res.stderr.strip()}")
                        continue
                    got = np.frombuffer(out.read_bytes(), F).reshape(expected.shape)
                    if got.tobytes() != expected.tobytes():
                        diff = np.abs(got - expected).max()
                        bad = int(np.sum(got.view(np.uint32) != expected.view(np.uint32)))
                        failures.append(f"{tag}: {bad} values differ (max abs {diff:.3g})")
    print(f"{checks} runs, {len(failures)} failures")
    for f in failures[:40]:
        print("  FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

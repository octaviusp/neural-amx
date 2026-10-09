"""Converter test: models built in PyTorch, Keras, MLX and ONNX (exported from PyTorch) are converted; the framework's
own output must match the reference (float32 tolerance) and the C runner must match the reference bit for bit.

    uv run --python 3.12 --with numpy --with torch --with tensorflow --with mlx --with onnx --with onnxscript \
        python -I tests/convert_test.py <work dir> <runner command>
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import neural_amx as nam  # noqa: E402

RNG = np.random.default_rng(7)
F = np.float32
results, failures = [], []


def check(name, model, framework_out, x, agents, runner, work):
    """framework_out: (rows, outputs) from the source framework for x (rows run in chunks of `agents`)."""
    ref = np.concatenate([r for r in _ref_chunks(model, x, agents, "f32")])
    err = float(np.abs(ref - framework_out).max())
    scale = float(np.abs(framework_out).max()) or 1.0
    ok = err <= 2e-5 * max(1.0, scale)
    results.append((name, err))
    if not ok:
        failures.append(f"{name}: framework vs reference max abs error {err:.3g}")
    path = work / f"{name}.safetensors"
    model.save(path)
    (work / f"{name}.in").write_bytes(np.array(x.shape, np.uint32).tobytes() + x.astype(F).tobytes())
    for prec, code in (("stored", 0), ("int8", 2), ("int4", 3)):
        expected = np.concatenate([r for r in _ref_chunks(model, x, agents, prec)])
        out = work / "out.bin"
        res = subprocess.run(runner + [str(path), str(work / f"{name}.in"), str(out), str(code), "0", "-1", str(agents)],
                             capture_output=True, text=True)
        if res.returncode:
            failures.append(f"{name}/{prec}: runner failed: {res.stderr.strip()}")
            continue
        got = np.frombuffer(out.read_bytes(), F).reshape(expected.shape)
        if got.tobytes() != expected.tobytes():
            failures.append(f"{name}/{prec}: C differs from the reference (max {np.abs(got - expected).max():.3g})")


def _ref_chunks(model, x, agents, prec):
    r = nam.Runner(model, prec)
    for i in range(0, len(x), agents):
        yield r.run(x[i:i + agents])


def torch_cases(runner, work):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
                        nn.Linear(64, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 8), nn.Softmax(dim=-1)).eval()
    bn = nn.Sequential(nn.Linear(20, 40), nn.BatchNorm1d(40), nn.LeakyReLU(0.1), nn.Linear(40, 30), nn.SiLU(),
                       nn.Linear(30, 10), nn.Tanh(), nn.Linear(10, 4), nn.Sigmoid())
    with torch.no_grad():
        bn[1].running_mean.uniform_(-1, 1)
        bn[1].running_var.uniform_(0.5, 2)
    bn.eval()
    conv = nn.Sequential(nn.Conv1d(4, 8, 5, stride=2, padding=2), nn.ReLU(), nn.Conv1d(8, 6, 3, padding="same"),
                         nn.Flatten(), nn.Linear(6 * 15, 3)).eval()

    class Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = nn.Linear(16, 32), nn.Linear(32, 32), nn.Linear(32, 5)

        def forward(self, x):
            x = Fn.relu(self.a(x))
            x = torch.sigmoid(self.b(x))
            return self.c(x)

    custom = Custom().eval()
    with torch.no_grad():
        for name, m, n_in in (("torch-mlp", mlp, 64), ("torch-bn", bn, 20), ("torch-custom", custom, 16)):
            x = RNG.standard_normal((13, n_in)).astype(F)
            check(name, nam.from_torch(m), m(torch.from_numpy(x)).numpy(), x, 1, runner, work)
        x = RNG.standard_normal((9, 4, 30)).astype(F)
        check("torch-conv", nam.from_torch(conv, inputs=120), conv(torch.from_numpy(x)).numpy(), x.reshape(9, -1), 1, runner, work)
        # recurrent: 5 agents, 6 steps; per-step layers [enc, tanh, gru, head]
        enc, gru, head = nn.Linear(20, 32), nn.GRU(32, 24, batch_first=True), nn.Linear(24, 6)
        seq = RNG.standard_normal((5, 6, 20)).astype(F)
        h = gru(torch.tanh(enc(torch.from_numpy(seq))))[0]
        want = head(h).numpy().transpose(1, 0, 2).reshape(-1, 6)  # step-major
        check("torch-gru", nam.from_torch([enc, nn.Tanh(), gru, head]), want, seq.transpose(1, 0, 2).reshape(-1, 20), 5, runner, work)
        # ONNX exported from PyTorch (the recurrent model as a whole sequence: steps must match the stateful runtime)
        class Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc, self.gru, self.head = enc, gru, head

            def forward(self, x):
                return self.head(self.gru(torch.tanh(self.enc(x)))[0])

        path = work / "onnx-gru.onnx"
        torch.onnx.export(Seq().eval(), (torch.zeros(5, 6, 20),), str(path), dynamo=False, opset_version=17)
        check("onnx-gru", nam.from_onnx(str(path)), want, seq.transpose(1, 0, 2).reshape(-1, 20), 5, runner, work)
        for name, m, shape in (("onnx-mlp", mlp, (1, 64)), ("onnx-conv", conv, (1, 4, 30)), ("onnx-bn", bn, (2, 20))):
            path = work / f"{name}.onnx"
            torch.onnx.export(m, (torch.zeros(shape),), str(path), dynamo=False, opset_version=17,
                              input_names=["x"], dynamic_axes={"x": {0: "n"}})
            x = RNG.standard_normal((7,) + shape[1:]).astype(F)
            check(name, nam.from_onnx(str(path)), m(torch.from_numpy(x)).numpy(), x.reshape(7, -1), 1, runner, work)


def keras_cases(runner, work):
    import keras
    from keras import layers as L
    keras.utils.set_random_seed(1)
    mlp = keras.Sequential([keras.Input((64,)), L.Dense(128, activation="relu"), L.LayerNormalization(),
                            L.Dense(64, activation="gelu"), L.Dropout(0.2), L.Dense(32, activation="silu"),
                            L.LeakyReLU(negative_slope=0.2), L.Dense(8, activation="softmax")])
    x = RNG.standard_normal((13, 64)).astype(F)
    check("keras-mlp", nam.from_keras(mlp), mlp(x, training=False).numpy(), x, 1, runner, work)
    conv = keras.Sequential([keras.Input((30, 4)), L.Conv1D(8, 5, strides=2, activation="relu"), L.Conv1D(6, 3, padding="same"),
                             L.BatchNormalization(), L.Flatten(), L.Dense(3)])
    bnl = conv.layers[2]
    bnl.moving_mean.assign(RNG.uniform(-1, 1, 6).astype(F))
    bnl.moving_variance.assign(RNG.uniform(0.5, 2, 6).astype(F))
    x = RNG.standard_normal((9, 30, 4)).astype(F)
    check("keras-conv", nam.from_keras(conv), conv(x, training=False).numpy(), x.reshape(9, -1), 1, runner, work)
    rnn = keras.Sequential([keras.Input((6, 20)), L.Dense(32, activation="tanh"), L.GRU(24, return_sequences=True), L.Dense(6)])
    seq = RNG.standard_normal((5, 6, 20)).astype(F)
    want = rnn(seq).numpy().transpose(1, 0, 2).reshape(-1, 6)
    check("keras-gru", nam.from_keras(rnn), want, seq.transpose(1, 0, 2).reshape(-1, 20), 5, runner, work)


def mlx_cases(runner, work):
    import mlx.core as mx
    import mlx.nn as nn
    from neural_amx.convert.mlx import Flatten
    mx.set_default_device(mx.cpu)  # the Metal GPU is less precise (see README); compare against MLX on CPU
    mx.random.seed(3)
    mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
                        nn.Linear(64, 32), nn.GELU("precise"), nn.Linear(32, 8), nn.Softmax())
    x = RNG.standard_normal((13, 64)).astype(F)
    check("mlx-mlp", nam.from_mlx(mlp), np.array(mlp(mx.array(x))), x, 1, runner, work)
    conv = nn.Sequential(nn.Conv1d(4, 8, 5, stride=2, padding=2), nn.ReLU(), nn.Conv1d(8, 6, 3, padding=1), Flatten(), nn.Linear(6 * 15, 3))
    x = RNG.standard_normal((9, 30, 4)).astype(F)  # MLX layout (N, L, C)
    check("mlx-conv", nam.from_mlx(conv, inputs=120, channels=4), np.array(conv(mx.array(x))), x.reshape(9, -1), 1, runner, work)
    enc, gru, head = nn.Linear(20, 32), nn.GRU(32, 24), nn.Linear(24, 6)
    seq = RNG.standard_normal((5, 6, 20)).astype(F)
    h = gru(mx.tanh(enc(mx.array(seq))))
    want = np.array(head(h)).transpose(1, 0, 2).reshape(-1, 6)
    check("mlx-gru", nam.from_mlx([enc, nn.Tanh(), gru, head]), want, seq.transpose(1, 0, 2).reshape(-1, 20), 5, runner, work)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    runner = shlex.split(sys.argv[2])
    for cases in (torch_cases, keras_cases, mlx_cases):
        try:
            cases(runner, work)
        except Exception as e:  # report and continue with the other frameworks
            import traceback
            failures.append(f"{cases.__name__}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    for name, err in results:
        print(f"  {name:14} framework vs reference max abs error {err:.2e}")
    print(f"{len(results)} models, {len(failures)} failures")
    for f in failures:
        print("  FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""QAT smoke test: a model prepared for int4 QAT in PyTorch and MLX, trained a few steps, must convert and the
runtime's int4 output must track the QAT forward pass (same quantizer up to activation rounding).
    uv run --python 3.12 --with numpy --with torch --with mlx python -I tests/qat_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import neural_amx as nam  # noqa: E402
from neural_amx import qat  # noqa: E402

rng = np.random.default_rng(0)
x = rng.standard_normal((256, 32)).astype(np.float32)
y = (x[:, :4].sum(1) > 0).astype(np.int64) + 2 * (x[:, 4:8].sum(1) > 0)
fails = []


def compare(name, qat_out, model):
    ref = nam.run(model, x, "int4")
    agree = float((ref.argmax(1) == qat_out.argmax(1)).mean())
    err = float(np.abs(ref - qat_out).max())
    print(f"{name}: argmax agreement {agree:.3f}, max abs diff {err:.3g}, int4 layers {[l.get('bits') for l in nam.quantize(model, 'int4').layers if 'bits' in l]}")
    if agree < 0.97:
        fails.append(name)


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
torch.manual_seed(0)
tm = qat.torch_prepare(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)), "int4")
opt = torch.optim.Adam(tm.parameters(), 1e-2)
for _ in range(100):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(tm(torch.from_numpy(x)), torch.from_numpy(y))
    loss.backward()
    opt.step()
with torch.no_grad():
    compare("torch", tm(torch.from_numpy(x)).numpy(), nam.from_torch(tm))

import mlx.core as mx  # noqa: E402
import mlx.nn as mnn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
mx.set_default_device(mx.cpu)
mm = qat.mlx_prepare(mnn.Sequential(mnn.Linear(32, 64), mnn.ReLU(), mnn.Linear(64, 64), mnn.ReLU(), mnn.Linear(64, 4)), "int4")
o = optim.Adam(learning_rate=1e-2)
step = mnn.value_and_grad(mm, lambda m, a, b: mnn.losses.cross_entropy(m(a), b, reduction="mean"))
for _ in range(100):
    _, g = step(mm, mx.array(x), mx.array(y))
    o.update(mm, g)
    mx.eval(mm.parameters())
compare("mlx", np.array(mm(mx.array(x))), nam.from_mlx(mm))
print("ok" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)

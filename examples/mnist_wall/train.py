"""MNIST wall: trains a digit classifier on *shot drawings* and exports it for neural_mnist.sma.

Players draw with bullet holes, so the input is a cloud of dots, not MNIST strokes. Each MNIST digit is turned into a
random set of "shots" sampled from its ink, then rendered with exactly the plugin's renderer (render() below and
render_canvas() in neural_mnist.sma): fit the dots into an 18-cell box, stamp a soft brush, re-center on the center of
mass. The network therefore sees in training what the plugin feeds it in game.

    uv run --python 3.12 --with torch --with numpy --with "neural-amx @ git+https://github.com/octaviusp/neural-amx" \
        python examples/mnist_wall/train.py <out dir>
Writes <out>/mnist.safetensors (int8) and <out>/painter_digits.txt (test drawings for the headless painter).
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import neural_amx as nam

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
BRUSH = 1.7      # a dot paints full intensity within 0.7 cells, fading to 0 at 1.7 cells
BOX = 18.0       # dots are fitted into an 18-cell box (the brush makes the ink about 20, like MNIST)
MAX_DOTS = 120
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_mnist(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + name, path)
        raw = gzip.decompress(path.read_bytes())
        out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28) if "x" in key else np.frombuffer(raw, np.uint8, offset=8)
    return out


GRID = torch.stack(torch.meshgrid(torch.arange(28.0) + 0.5, torch.arange(28.0) + 0.5, indexing="xy"), -1).reshape(784, 2)


def splat(p, mask):
    d = torch.cdist(p, GRID.to(p.device))                     # (B, K, 784)
    v = torch.clamp(BRUSH - d, 0.0, 1.0) * mask[..., None]
    return v.amax(1)                                           # brush stamps combine with max


def render(points, mask):
    """points (B, K, 2) as (column, row) in canvas cells, mask (B, K). Mirrors render_canvas() in the plugin."""
    big = 1e9
    x, y = points[..., 0], points[..., 1]
    xmin, xmax = torch.where(mask, x, big).amin(1), torch.where(mask, x, -big).amax(1)
    ymin, ymax = torch.where(mask, y, big).amin(1), torch.where(mask, y, -big).amax(1)
    s = BOX / torch.clamp(torch.maximum(xmax - xmin, ymax - ymin), min=3.0)
    c = torch.stack([(xmin + xmax) / 2, (ymin + ymax) / 2], -1)
    p = (points - c[:, None]) * s[:, None, None] + 14.0
    img = splat(p, mask)
    total = img.sum(1).clamp(min=1e-6)
    com = (img[..., None] * GRID.to(p.device)).sum(1) / total[:, None]
    return splat(p + (14.0 - com)[:, None], mask)


def sample_shots(images, rng, augment=True, k_range=(20, 100)):
    """Random dots on each digit's ink, as a player would shoot it."""
    B = len(images)
    pts = np.zeros((B, MAX_DOTS, 2), np.float32)
    mask = np.zeros((B, MAX_DOTS), bool)
    for i, im in enumerate(images):
        rows, cols = np.nonzero(im > 90)
        k = rng.integers(*k_range)
        idx = rng.choice(len(rows), k, replace=len(rows) < k)
        p = np.stack([cols[idx] + 0.5, rows[idx] + 0.5], -1) + rng.uniform(-0.5, 0.5, (k, 2))
        if augment:
            a = np.deg2rad(rng.uniform(-12, 12))
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            p = (p - 14) @ rot.T * [rng.uniform(0.8, 1.2), 1.0] + 14
            if rng.random() < 0.1:  # a stray shot near the drawing
                p = np.vstack([p, p[rng.integers(k)] + rng.uniform(-3, 3, 2)])
        n = min(len(p), MAX_DOTS)
        pts[i, :n], mask[i, :n] = p[:n], True
    return torch.from_numpy(pts), torch.from_numpy(mask)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/mnist")
    out.mkdir(parents=True, exist_ok=True)
    data = load_mnist(out / "data")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=12 * (60000 // 256))
    xtr, ytr = data["train_x"], torch.from_numpy(data["train_y"].astype(np.int64))
    for epoch in range(12):
        order = rng.permutation(60000)
        for i in range(0, 60000 - 255, 256):
            idx = order[i:i + 256]
            p, m = sample_shots(xtr[idx], rng)
            x = render(p.to(DEV), m.to(DEV))
            loss = nn.functional.cross_entropy(model(x), ytr[idx].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}", flush=True)

    model = model.cpu().eval()
    test_rng = np.random.default_rng(1)
    p, m = sample_shots(data["test_x"][:5000], test_rng, augment=False)
    xt = render(p, m).numpy()
    yt = data["test_y"][:5000]
    exported = nam.from_torch(model, labels=[str(d) for d in range(10)])
    exported.layers.append({"op": "softmax"})
    for prec in ("f32", "int8", "int4"):
        acc = (nam.run(exported, xt, prec).argmax(1) == yt).mean()
        print(f"shot-drawn test digits, {prec}: {acc:.4f}")
    nam.quantize(exported, "int8").save(out / "mnist.safetensors")

    # painter drawings: 3 test digits per class, 60 shots each, in canvas cells (0..28)
    lines, chosen = [], []
    for d in range(10):
        chosen += list(np.nonzero(data["test_y"][5000:] == d)[0][:3] + 5000)
    p, m = sample_shots(data["test_x"][chosen], np.random.default_rng(2), augment=False, k_range=(60, 61))
    pred = nam.run(exported, render(p, m).numpy(), "int8").argmax(1)
    for j, i in enumerate(chosen):
        dots = p[j][m[j]].numpy()
        lines.append(f"{data['test_y'][i]} {pred[j]} {len(dots)} " + " ".join(f"{v:.3f}" for v in dots.reshape(-1)))
    (out / "painter_digits.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'mnist.safetensors'} and {len(lines)} painter drawings")


if __name__ == "__main__":
    main()

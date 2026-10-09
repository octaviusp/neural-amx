"""Example: train a tiny bot policy in PyTorch with int4 QAT and export it for AMX Mod X.

    uv run --with torch --with numpy --with /path/to/neural-amx python examples/train_torch.py policy.safetensors

The features and labels here are synthetic; record yours from the game with the same feature code the plugin uses.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

import neural_amx as nam
from neural_amx import qat

ACTIONS = ["attack", "strafe_left", "strafe_right", "retreat", "reload", "hold", "flank", "chase"]
FEATURES = 64

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (20000, FEATURES)).astype(np.float32)
teacher = rng.standard_normal((FEATURES, len(ACTIONS))).astype(np.float32)
y = np.argmax(np.tanh(x @ teacher), axis=1)

policy = nn.Sequential(nn.Linear(FEATURES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(ACTIONS)))
opt = torch.optim.Adam(policy.parameters(), 1e-3)
xt, yt = torch.from_numpy(x), torch.from_numpy(y)


def train(epochs):
    for _ in range(epochs):
        for i in range(0, len(x), 256):
            opt.zero_grad()
            nn.functional.cross_entropy(policy(xt[i:i + 256]), yt[i:i + 256]).backward()
            opt.step()


train(10)                          # float training
qat.torch_prepare(policy, "int4")  # then fine-tune with the runtime's int4 quantizer
train(3)

model = nam.from_torch(policy.eval(), labels=ACTIONS)
model.layers.append({"op": "softmax"})  # probabilities for neural_sample
nam.quantize(model, "int4").save(sys.argv[1] if len(sys.argv) > 1 else "policy.safetensors")
acc = (nam.run(model, x[:2000], "int4").argmax(1) == y[:2000]).mean()
print(f"int4 accuracy on training samples: {acc:.3f}")

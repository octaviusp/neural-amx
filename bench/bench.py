"""Benchmarks the C runtime: model sizes x precisions x kernels x batching, for each runner command.

    uv run --no-project --with numpy python -I bench/bench.py <work dir> <runner command>...
    (e.g. build/runner-native  "node build/runner-wasm-simd.js")
Writes <work dir>/bench.json and prints ns per inference (best of 5 rounds over 64 samples).
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import neural_amx as nam  # noqa: E402

SIZES = {"S 64-128-128-8": [64, 128, 128, 8], "M 256-512-512-32": [256, 512, 512, 32],
         "L 1024-1024-1024-64": [1024, 1024, 1024, 64]}
PRECISIONS = {"f32": 1, "int8": 2, "int4": 3}
KERNELS = {"dot": 1, "acc16": 2}


def make(work: Path):
    rng = np.random.default_rng(0)
    paths = {}
    for name, dims in SIZES.items():
        t, layers = {}, []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            t[f"fc{i}.weight"] = (rng.standard_normal((b, a)) / np.sqrt(a)).astype(np.float32)
            t[f"fc{i}.bias"] = np.zeros(b, np.float32)
            layers.append({"op": "dense", "weight": f"fc{i}.weight", "bias": f"fc{i}.bias"})
            if i < len(dims) - 2:
                layers.append({"op": "relu"})
        p = work / f"bench-{name.split()[0]}.safetensors"
        nam.Model(layers, t, dims[0]).save(p)
        paths[name] = p
    return paths


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    runners = sys.argv[2:]
    paths = make(work)
    rows = []
    for runner in runners:
        for size, path in paths.items():
            for prec, pc in PRECISIONS.items():
                for kname, kc in KERNELS.items():
                    if prec == "f32" and kname == "acc16":
                        continue
                    for batch in (0, 1):
                        cmd = shlex.split(runner) + ["--bench", str(path), str(pc), str(kc), str(batch), "64", "60"]
                        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
                        rows.append({"runner": runner, "backend": res["backend"], "model": size, "precision": prec,
                                     "kernel": kname if prec != "f32" else "-", "batch": batch, "ns": res["ns"]})
                        print(f"{res['backend']:13} {size:22} {prec:5} {rows[-1]['kernel']:5} batch={batch}  {res['ns']:10.1f} ns")
    (work / "bench.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

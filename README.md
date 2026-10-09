# neural-amx

Fast, deterministic neural-network inference for **AMX Mod X** (and any C program). Train in **PyTorch**,
**TensorFlow/Keras**, **MLX** or anything that exports **ONNX**; ship one `.safetensors` file; run it from Pawn in about
a microsecond per decision.

```pawn
#include <neural_amx>

new Neural:g_policy;
public plugin_init() { g_policy = neural_load("neural/policy.safetensors", NEURAL_INT4); }

decide(id) {
    new Float:features[64], Float:actions[8];
    build_features(id, features);
    return neural_run(g_policy, features, actions, .slot = id); // index of the best action
}
```

- **Self-contained C99 core**: no dependencies; safetensors and JSON parsing included; one file per concern.
- **Fast everywhere**: WebAssembly SIMD128, x86 SSE2 and ARM NEON kernels, int8/int4 weights, batched inference, and a
  load-time **autotune** that picks the fastest kernel on the host it runs on.
- **Deterministic**: identical output bits on every backend (wasm, x86, i386 HLDS, ARM, scalar) and in the Python
  reference implementation. No libm transcendental functions, no fused multiply-add.
- **Safe from Pawn**: every array is bounds-checked against the plugin's memory; errors are logged, not crashes.

## Quick start

1. **Convert** (Python ≥ 3.10):
   ```sh
   pip install "neural-amx[torch] @ git+https://github.com/octaviusp/neural-amx"   # or [tensorflow], [mlx], [onnx]
   ```
   ```python
   import neural_amx as nam
   model = nam.from_torch(torch_model, labels=["attack", "hold", "retreat"])   # nn.Sequential or a simple module
   nam.quantize(model, "int4").save("policy.safetensors")
   ```
   or from the command line:
   ```sh
   neural-amx convert model.onnx -o policy.safetensors --precision int4
   neural-amx convert state_dict.safetensors --spec "input(64) dense(fc1) relu dense(fc2)" -o policy.safetensors
   ```
2. **Install** the module (`neural_amx_amxx_i386.so`, or the wasm side module) in `addons/amxmodx/modules/` and the
   model in `addons/amxmodx/data/neural/`; include `amxx/neural_amx.inc` in your plugin.
3. **Run** it: `neural_run`, or `neural_run_batch` for many agents in one call. See `examples/`.

## Converters

| Source | API | Notes |
|---|---|---|
| PyTorch | `from_torch(module, inputs=None, labels=None)` | `nn.Sequential`, lists, or modules whose `forward` is a straight chain (traced with `torch.fx`) |
| TensorFlow / Keras 3 | `from_keras(model)` | Sequential or straight functional models; channels-last Conv1D handled automatically |
| MLX | `from_mlx(model, inputs=None, channels=None)` | `nn.Sequential` or lists; put `neural_amx.convert.mlx.Flatten()` after Conv1d blocks |
| ONNX | `from_onnx(path)` | from `torch.onnx.export`, `tf2onnx`, `skl2onnx`...; decomposed activations recognized |
| Plain safetensors | `from_safetensors(path, spec)` / `neural_load(file, precision, spec)` | any PyTorch `state_dict`, described by a one-line spec |

**Layers**: dense/linear, conv1d (stride, zero padding), GRU (PyTorch semantics, one step per call, one hidden state per
agent slot), layernorm, batchnorm, affine/normalization, relu, leaky_relu, sigmoid, tanh, gelu (erf), gelu_tanh,
silu/swish, softmax, flatten, dropout/identity.

**Spec grammar**: `input(N)`, `dense(prefix)`, `conv1d(prefix, stride=2, padding=1)`, `gru(prefix)` (PyTorch
`weight_ih_l0` names), `layernorm(prefix, eps=1e-5)`, `batchnorm(prefix)`, `leaky_relu(0.2)` and the activation names.

## Precision and kernels

| Precision | Weights | Activations | Use it when |
|---|---|---|---|
| `NEURAL_F32` | float32 | float32 | small networks (≤ ~30k multiply-adds) where everything costs ~1 µs anyway |
| `NEURAL_INT8` | int8 per row | int8 per sample (dynamic) | lossless without retraining in most models |
| `NEURAL_INT4` | int4 hidden, int8 first/last and recurrent | int8 per sample | **default for wide layers**: 1/5 of the memory, fastest on M/L sizes; train with QAT (`neural_amx.qat`) |
| `NEURAL_STORED` | as the file stores it | | files written by `nam.quantize` |

Quantizing at load time needs no calibration: activations are scaled per sample. `nam.quantize` stores the same weights
the loader would compute, so a stored int4 file and an int4 load of the float file give identical results.

**Kernels** (chosen by the load-time autotune, ~10-30 ms; the result is the same either way):

- `acc16` accumulates 4-bit x 8-bit products in int16 lanes (up to 36 before widening, exact by construction). It wins
  on **ARM** (Apple Silicon, ARM servers, browsers on ARM), where the int16 dot product is several instructions.
- `dot` uses int16 pair dot products into int32 (`pmaddwd`, one instruction on **x86**). It wins on x86 servers.
- `batch` runs four samples per pass over the weights; it pays off from medium-sized layers.

### Measured (ns per inference, best configuration per row; 2026-10)

| Model (multiply-adds) | Apple M5 Pro, wasm SIMD128 in Node | same, native NEON | x86 VPS (i9-14900K VM, 2 vCPU), wasm in Node | plain Pawn |
|---|---|---|---|---|
| 64-128-128-8 (25.6k) | 940 (f32) · 960 (int4 acc16) | 842 (int4 acc16 batch) | 3 949 (f32 batch) | ~2 100 000 |
| 256-512-512-32 (410k) | 10 046 (int4 acc16 batch) · 13 310 (f32 batch) | 9 579 | 44 551 (int8 dot batch) · 56 383 (f32 batch) | - |
| 1024-1024-1024-64 (2.16M) | 53 257 (int4 acc16 batch) · 70 493 (f32 batch) | 52 624 | 221 328 (int4 dot batch) · 275 020 (f32 batch) | - |

Inside a real Counter-Strike 1.6 server (Xash3D + ReGameDLL + AMX Mod X in WebAssembly, 10 bots on de_dust2):
`neural_run` costs ~1.1 µs for the small model; building 64 game features in Pawn (positions, nearest enemies,
visibility traces) costs ~19 µs per bot, so **feature code, not the network, is the budget to watch**. The VPS runs
were noisy (shared machine, ±20-30%); `bench/bench.py` reproduces every number on your hardware.

## Determinism and tests

- `tests/golden.py` builds models covering every op (odd sizes, conv1d, GRU sequences, batchnorm via spec, stored int8
  and int4, zero and large inputs) and requires every runner to match `python/neural_amx/reference.py` **bit for bit**
  for every precision, kernel and batch mode. Verified: macOS NEON, scalar, x86_64 SSE2, wasm SIMD128, wasm scalar,
  Linux x86_64 and Linux i386 (SSE2). An i386 x87 build fails it: build HLDS modules with `-msse2 -mfpmath=sse`.
- `tests/convert_test.py` converts PyTorch, Keras, MLX and ONNX models (MLP, conv1d, GRU, batchnorm, custom forward);
  the framework's own output matches the reference within ~1e-6 and the C runtime matches the reference exactly.
  MLX on the Metal GPU differs from its own CPU result by up to ~5e-4; compare against `mx.cpu`.
- `tests/qat_test.py`: int4 QAT in PyTorch and MLX tracks the runtime's int4 output (100% argmax agreement).

```sh
EMSDK_ENV=/path/to/emsdk_env.sh tests/build-runners.sh
python tests/golden.py build/golden build/runner-native build/runner-scalar "node build/runner-wasm-simd.js"
python tests/convert_test.py build/convert build/runner-native     # needs torch, tensorflow, mlx, onnx
python bench/bench.py build/bench build/runner-native "node build/runner-wasm-simd.js"
```

## Building the AMX Mod X module

```sh
export AMXX_SDK=/path/to/amxmodx HLSDK=/path/to/hlsdk METAMOD=/path/to/metamod-hl1
amxx/build.sh linux   # HLDS / ReHLDS: build/amxx/neural_amx_amxx_i386.so (gcc -m32, SSE2)
amxx/build.sh wasm    # Emscripten side module for WebAssembly servers (EMSDK_ENV; WASM_FLAGS to match your server)
amxx/build.sh host    # compile check on this machine
```

Models load from `<amxx_datadir>/<file>`; handles are freed on map change. Recurrent models keep one hidden state per
slot (default 33: use the player id); `neural_reset(model, id)` on respawn.

## File format

A standard safetensors file. The metadata key `neural_amx` holds the graph as JSON:
`{"format":1,"inputs":64,"labels":[...],"layers":[{"op":"dense","weight":"fc1.weight","bias":"fc1.bias"},{"op":"relu"},...]}`.
Quantized weights are `I8` (`[out, in]`) or packed int4 as `U8` (`[out, ceil(in/2)]`, low nibble first) with a float32
`<weight>.scale` per row and `"bits"` on the layer.

## Layout

| Path | Contents |
|---|---|
| `core/` | C99 runtime: `nam.h` (API), kernels, runner, loader, safetensors/JSON parsers, SIMD and math headers |
| `amxx/` | AMX Mod X module, `neural_amx.inc`, build script |
| `python/neural_amx/` | format, reference (the specification), quantize, QAT, converters, CLI |
| `tests/`, `bench/`, `examples/` | bit-exactness and converter tests, benchmarks, a training script and a bot plugin |

## License

MIT (see `LICENSE`). The AMX Mod X SDK files you compile the module with keep their own license.

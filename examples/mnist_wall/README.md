# MNIST wall: draw a digit with bullets, a neural network reads it

`say /mnist` opens a 28x28 canvas on the wall you are looking at (a green frame). Shoot a digit inside it; the HUD keeps
the network's top 3 on screen while you draw, the server console prints each reading
(`[MNIST] Player PREDICTED: 7 (93.1%), 1 (4.0%), 9 (1.2%) | 14 shots`), and `say /mnist ok` announces the answer in the chat:
`[MNIST] Player drew a 7 (93% sure).` `say /mnist clear` erases, `say /mnist off` closes.

## How it works

1. **Shots become dots.** `Ham_TraceAttack` on `worldspawn`/`func_wall` gives the exact point each bullet hits (where the
   decal appears). `mnist_mode 1` uses the crosshair point instead (ignores spread and recoil).
2. **Dots become an MNIST-like image.** The dots are fitted into an 18-cell box, stamped with a soft brush (full
   intensity within 0.7 cells, zero at 1.7) and re-centered on the center of mass, like MNIST's own preprocessing.
3. **The network** (784-256-128-10 MLP, int8, ~470 KB resident) runs through neural-amx in about 1 µs-scale per layer
   of work; the Pawn renderer is the expensive part and still runs per shot.

Bullet holes are dots, not pen strokes, so `train.py` trains on *shot drawings*: each MNIST digit becomes 20-100 random
dots sampled from its ink (rotation, width and stray-shot augmentation), rendered by exactly the plugin's renderer.

## Ready-made model

`mnist.safetensors` (int8, 239 KB, sha256 `8c23c4fb105117ad9d8e30410fa52c778c410cc6bf59b90a66030aa6c7cef312`) is the output
of `train.py` as committed. Its weights are derived from the MNIST database (Yann LeCun, Corinna Cortes, Christopher
J. C. Burges), distributed under CC BY-SA 3.0; the model file is shared under the same license. Code in this folder is MIT.

## Run it

```sh
uv run --python 3.12 --with torch --with numpy --with "neural-amx @ git+https://github.com/octaviusp/neural-amx" \
    python examples/mnist_wall/train.py build/mnist        # ~4 min on Apple Silicon (MPS); downloads MNIST once
```
Copy `mnist.safetensors` (or your own `build/mnist/mnist.safetensors`) to `addons/amxmodx/data/neural/`, install the neural-amx module, compile
`neural_mnist.sma` with `amxx/neural_amx.inc`, and add it to `plugins.ini`.

| cvar | default | meaning |
|---|---|---|
| `mnist_model` | `neural/mnist.safetensors` | model under the data directory |
| `mnist_cell` | `4.0` | units per canvas cell (the canvas is 28 cells wide) |
| `mnist_mode` | `0` | 0: bullet impacts, 1: crosshair point |
| `mnist_live` | `1` | top 3 on the HUD (kept on screen while the canvas is open) and a `PREDICTED:` line in the server console after each burst of shots |
| `mnist_debug` | `0` | print every shot's canvas position and whether it was kept |

Admin/test commands: `mnist_open <id>`, `mnist_guess <id>`, `mnist_clear <id>`, `mnist_status` (the loaded model); other plugins can call the public
functions `mnist_api_open/clear/close/guess/dots(id)` with `callfunc`.

## Measured

Shot-drawn MNIST test digits (5,000, 20-100 dots each, no augmentation): 93.8% float32, 93.9% int8, 93.9% int4.
Headless in a real Counter-Strike 1.6 server (Xash3D + ReGameDLL + AMX Mod X in WebAssembly), a plugin-driven bot shot
30 test digits at a de_dust2 wall with an M4A1; see the csuniverse `tests/neural-mnist` harness for the results.

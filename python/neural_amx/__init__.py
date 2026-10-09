"""neural-amx: train anywhere (PyTorch, TensorFlow/Keras, MLX, ONNX), run fast deterministic inference in
AMX Mod X (and any C program) from a single safetensors file.

    import neural_amx as nam
    model = nam.from_torch(torch_model, inputs=64, labels=["attack", "retreat"])
    nam.quantize(model, "int4").save("policy.safetensors")
    nam.run(model, x)            # bit-exact reference of the C runtime
"""
from .format import Model, load, read_safetensors, save, write_safetensors
from .quantize import quantize
from .reference import Runner, run

__version__ = "1.0.0"
__all__ = ["Model", "Runner", "load", "save", "run", "quantize", "read_safetensors", "write_safetensors",
           "from_torch", "from_keras", "from_mlx", "from_onnx", "from_safetensors", "from_spec"]


def __getattr__(name):  # converters import their framework lazily
    if name == "from_torch":
        from .convert.torch import from_torch
        return from_torch
    if name == "from_keras":
        from .convert.keras import from_keras
        return from_keras
    if name == "from_mlx":
        from .convert.mlx import from_mlx
        return from_mlx
    if name == "from_onnx":
        from .convert.onnx import from_onnx
        return from_onnx
    if name in ("from_safetensors", "from_spec"):
        from .convert import spec
        return getattr(spec, name)
    raise AttributeError(name)

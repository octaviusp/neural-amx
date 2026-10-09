// neural-amx: fast, deterministic neural-network inference in portable C99.
// Loads safetensors models (graph in the metadata, or any PyTorch-style state_dict plus a one-line spec) and runs them
// in float32, int8 or int4 with per-host kernel autotuning. Same bits on WebAssembly, x86, ARM and scalar builds.
// SPDX-License-Identifier: MIT
#ifndef NAM_H
#define NAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAM_VERSION "1.0.0"
#define NAM_BATCH 4 // samples per weight pass in batched inference

typedef struct nam_model nam_model;

typedef enum {
  NAM_PRECISION_STORED = 0, // as the file stores it (float or quantized)
  NAM_PRECISION_F32 = 1,    // float32 everywhere (quantized files are dequantized)
  NAM_PRECISION_INT8 = 2,   // int8 weights, dynamic int8 activations
  NAM_PRECISION_INT4 = 3,   // int4 weights in hidden layers, int8 in the first/last weighted layer and recurrences
} nam_precision;

typedef enum {
  NAM_KERNEL_AUTO = 0,  // time the choices at load and keep the fastest (the result is identical either way)
  NAM_KERNEL_DOT = 1,   // i32 pair dot products (fastest on x86)
  NAM_KERNEL_ACC16 = 2, // int16-lane accumulation where exact (fastest on arm64 for int4 layers)
} nam_kernel;

typedef struct {
  nam_precision precision;
  nam_kernel kernel;
  int batch; // -1 auto, 0 off, 1 on
  int slots; // recurrent state slots (one per agent); default 33
} nam_options;

void nam_default_options(nam_options *options);

// `data`/`size`: a whole safetensors file. `spec`: NULL to use the graph stored in the file's metadata, or a layer
// list for a plain state_dict, e.g. "input(64) dense(fc1) relu dense(fc2) softmax". Returns NULL and writes a
// message to `err` on failure. The model copies what it needs; `data` can be freed after the call.
nam_model *nam_load(const void *data, size_t size, const char *spec, const nam_options *options, char *err,
                    size_t errlen);
void nam_free(nam_model *model);

int nam_inputs(const nam_model *model);
int nam_outputs(const nam_model *model);

// Runs `count` samples: `in` is count x inputs floats, `out` count x outputs. Sample i uses recurrent state slot
// first_slot + i. Returns count, or -1 when the slots are out of range.
int nam_run(nam_model *model, const float *in, int count, float *out, int first_slot);
void nam_reset(nam_model *model, int slot); // clear recurrent state; slot -1 clears all

int nam_tune(nam_model *model); // re-times the kernel choices on this host; returns 1
void nam_set_kernel(nam_model *model, int acc16, int batch); // explicit choice (benchmarks, tests)
int nam_describe(const nam_model *model, char *buf, size_t len);
const char *nam_label(const nam_model *model, int index); // output label from the metadata, or NULL
size_t nam_weight_bytes(const nam_model *model);           // resident weight memory
const char *nam_backend(void);                             // "wasm-simd128", "sse2", "neon" or "scalar"

// helpers
int nam_argmax(const float *v, int n);
void nam_softmax(float *v, int n, float temperature);
int nam_sample(const float *probabilities, int n, float uniform01);

#ifdef __cplusplus
}
#endif

#endif // NAM_H
